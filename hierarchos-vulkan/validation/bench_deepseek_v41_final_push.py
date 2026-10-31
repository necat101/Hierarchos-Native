"""Compare final V4.1 resident/streamed compute optimizations against a frozen probe.

Old probes divided the median of individual decode calls by --decode again.
Read the phase records directly so that both old and corrected probes use the
same unit. A final-row digest is a performance guard; full-logit CPU tests are
a separate requirement for qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    assert args.repeats >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    hashes = {leg: sha(path) for leg, path in binaries.items()}
    results = {}
    for residency in ['resident', 'default']:
        legs = {'baseline': [], 'candidate': []}
        for repeat in range(args.repeats):
            order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
            for leg in order:
                run = subprocess.run([str(binaries[leg]), '--model', str(args.model.resolve()),
                    '--prefill', '8', '--decode', '8', '--steps', '3', '--warmup', '1',
                    '--residency', residency], cwd=ROOT, text=True, capture_output=True)
                name = args.output / f'{residency}-{repeat}-{leg}'
                name.with_suffix('.out').write_text(run.stdout)
                name.with_suffix('.log').write_text(run.stderr)
                assert run.returncode == 0, (leg, residency, run.stderr[-2000:])
                summary = dict(re.findall(r'(\w+)=([^\s]+)', run.stdout))
                digests = re.findall(r'v41_probe_logits .*digest=([0-9a-f]+)', run.stderr)
                assert len(digests) == 4 and set(digests) == {'90e096ebc5bf0659'}
                record = {}
                load = re.search(r'v41_probe_load ms=([0-9.]+) submissions=(\d+)', run.stderr)
                assert load
                record['load_ms'] = float(load[1])
                record['load_submissions'] = int(load[2])
                io = {phase: json.loads(value) for phase, value in re.findall(r'v41_probe_io phase=(\w+) json=(.*)', run.stderr)}
                if leg == 'candidate':
                    assert set(io) == {'warmup_complete', 'complete'}
                    end, start = io['complete'], io['warmup_complete']
                    assert end['cache_peak_bytes'] <= end['capacity_bytes'] <= 32 * 1024 * 1024
                    record['io'] = io
                    record['measured_payload_reads'] = end['payload_reads'] - start['payload_reads']
                    record['measured_cache_hits'] = end['cache_hits'] - start['cache_hits']
                for phase, count in [('prefill', 3), ('decode', 24)]:
                    samples = re.findall(rf'v41_probe_phase phase={phase} .*measured=true ms=([0-9.]+).*submissions=(\d+)', run.stderr)
                    assert len(samples) == count, (phase, len(samples))
                    record[phase + '_ms'] = statistics.median(float(ms) for ms, _ in samples)
                    record[phase + '_submissions'] = statistics.median(int(n) for _, n in samples)
                if leg == 'candidate':
                    assert abs(float(summary['decode_ms_per_token']) - record['decode_ms']) <= 0.0011
                record['reported_decode_ms_per_token'] = float(summary['decode_ms_per_token'])
                record['digest'] = summary['digest']
                legs[leg].append(record)
        result = dict(runs=legs, digest='90e096ebc5bf0659')
        for metric in ['load_ms', 'load_submissions', 'prefill_ms', 'decode_ms', 'prefill_submissions', 'decode_submissions']:
            result[metric] = {leg: statistics.median(r[metric] for r in rows) for leg, rows in legs.items()}
        result['prefill_speedup'] = result['prefill_ms']['baseline'] / result['prefill_ms']['candidate']
        result['decode_speedup'] = result['decode_ms']['baseline'] / result['decode_ms']['candidate']
        assert result['decode_submissions']['candidate'] == result['decode_submissions']['baseline']
        assert result['load_submissions']['candidate'] == result['load_submissions']['baseline']
        assert result['prefill_speedup'] > 1 and result['decode_speedup'] > 1
        results[residency] = result
        print(residency, 'raw decode ms/token', result['decode_ms'],
              'submissions', result['decode_submissions'], 'speedup', result['decode_speedup'], flush=True)
        assert all(sha(path) == hashes[leg] for leg, path in binaries.items())
        (args.output / 'results.json').write_text(json.dumps(dict(binary_hashes=hashes,
            timing_unit='one decode call containing one token', repeats=args.repeats,
            steps=3, warmup=1, prefill_tokens=8, decode_tokens=8, results=results,
            limits=['final-row digest guard; full-logit CPU gates required separately',
                    'scaled 40-layer/384-expert fixture, vocab 67; no production throughput claim']), indent=2) + '\n')


if __name__ == '__main__':
    main()

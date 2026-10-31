"""Alternating, frozen-binary A/B for warm parameter streaming.

Timings describe the supplied build profile, not an inferred production speedup.
Each leg also records I/O, submissions, buffer peaks, exact logits and (for
training) the uninterrupted loss-sequence digest. CPU qualification is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-profile', required=True)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    assert args.iterations >= 2 and args.repeats >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((args.models / 'fixture.json').read_text())
    fixtures = {}
    for label, batch, seq in [('prefill', 1, 8), ('batch4', 4, 8), ('decode', 1, 1)]:
        data = dict(fixture, batch_size=batch, seq_len=seq)
        for key in ['input_ids', 'attention_mask']:
            data[key] = (fixture[key] * batch)[:batch * seq]
        path = args.output / (label + '.json')
        path.write_text(json.dumps(data))
        fixtures[label] = path
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    binary_hashes = {leg: sha(path) for leg, path in binaries.items()}
    results = []
    cases = [('dense-prefill', 'llama-h512-e0', 'prefill', 'inference'),
             ('dense-batch4', 'llama-h512-e0', 'batch4', 'inference'),
             ('dense-decode', 'llama-h512-e0', 'decode', 'inference'),
             ('dense-multitile', 'llama-h1024-i3072', 'prefill', 'inference'),
             ('moe-prefill', 'mixtral-h512-e8', 'prefill', 'inference'),
             ('dense-training', 'llama-h512-e0', 'prefill', 'train'),
             ('moe-training', 'mixtral-h512-e8', 'prefill', 'train')]
    for name, model, fixture, mode in cases:
        legs = {'baseline': [], 'candidate': []}
        reference = None
        for repeat in range(args.repeats):
            order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
            for leg in order:
                command = [str(binaries[leg]), str(args.models / model), str(fixtures[fixture]),
                           '2', str(args.iterations), mode]
                run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                stem = args.output / f'{name}-{repeat}-{leg}'
                stem.with_suffix('.stderr').write_text(run.stderr)
                assert run.returncode == 0, (name, leg, run.stderr[-3000:])
                report = json.loads(run.stdout)
                stem.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
                if reference is None:
                    reference = report
                assert report['sequence_sha256'] == reference['sequence_sha256'], (name, leg, 'sequence')
                assert report['logits'] == reference['logits'], (name, leg, 'logits')
                report['file_reads'] = sum(row['payload_reads'] for _, row in report['io_after']) - sum(row['payload_reads'] for _, row in report['io_before'])
                report['file_bytes'] = sum(row['payload_bytes_read'] for _, row in report['io_after']) - sum(row['payload_bytes_read'] for _, row in report['io_before'])
                assert all(row['cache_peak_bytes'] <= 32 * 1024 * 1024 for _, row in report['io_after'])
                legs[leg].append(report)
        medians = {leg: statistics.median(r['median_ms'] for r in rows) for leg, rows in legs.items()}
        reads = {leg: statistics.median(r['file_reads'] for r in rows) for leg, rows in legs.items()}
        result = dict(case=name, mode=mode, median_ms=medians, speedup=medians['baseline']/medians['candidate'],
                      median_file_reads=reads, bit_exact=True, logit_values=len(reference['logits']),
                      sequence_sha256=reference['sequence_sha256'])
        results.append(result)
        print(f"{name}: {result['speedup']:.3f}x, medians={medians}, reads={reads}, bit-exact", flush=True)
        (args.output / 'results.json').write_text(json.dumps(dict(build_profile=args.build_profile,
            binary_hashes=binary_hashes, results=results, repeats=args.repeats,
            iterations=args.iterations, warmup=2, strict_atol=2e-7, rtol=0), indent=2) + '\n')


if __name__ == '__main__':
    main()

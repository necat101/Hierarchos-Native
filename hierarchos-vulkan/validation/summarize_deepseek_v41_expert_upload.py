"""Bind expert upload/cache measurements to CPU logit and PEFT regression evidence."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def logits(path):
    records = defaultdict(list)
    for line in path.read_text().splitlines():
        if 'V4.1' in line:
            line = line[line.index('V4.1'):]
        m = re.search(r'^(.*?)\b(\d+) logits, drift ([0-9.eE+\-]+)', line)
        if not m:
            m = re.search(r'^(.*?\blogits \d+: )(\d+) values, worst ([0-9.eE+\-]+)', line)
        if m:
            records[(m[1], int(m[2]))].append(float(m[3]))
    assert records, path
    return records


def summary(records):
    result = dict(comparisons=sum(map(len, records.values())),
        values=sum(key[1] * len(values) for key, values in records.items()),
        max_abs=max(value for values in records.values() for value in values))
    assert result['max_abs'] < 2e-7, result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    evidence = args.evidence
    performance_path = evidence / 'ab-release/results.json'
    performance = json.loads(performance_path.read_text())
    assert sha(evidence / 'before/probe.exe') == performance['binary_hashes']['baseline']
    assert sha(evidence / 'candidate-probe.exe') == performance['binary_hashes']['candidate']
    assert set(performance['results']) == {'resident', 'default'}
    for row in performance['results'].values():
        assert row['digest'] == '90e096ebc5bf0659'
        assert row['decode_submissions']['candidate'] <= row['decode_submissions']['baseline']
    required = {'vulkan-regression': 38, 'generic-streaming': 3, 'offload-regression': 16,
                'v41-default-gate': 185, 'package-default': 1, 'joint-resident': 1, 'joint-all-streamed': 1,
                'v41-uninterrupted': 1, 'cli-tests': 41}
    tests = {}
    for name, expected in required.items():
        path = evidence / (name + '.log')
        match = re.search(r'test result: ok\. (\d+) passed; 0 failed;', path.read_text())
        assert match and int(match[1]) == expected, name
        tests[name] = dict(**binding(path), passed=expected)
    numeric = {name: summary(logits(evidence / (name + '.log'))) for name in
               ['v41-default-gate', 'package-default', 'joint-resident', 'joint-all-streamed', 'v41-uninterrupted']}
    assert numeric['package-default']['values'] == 3216
    assert numeric['joint-resident']['values'] == numeric['joint-all-streamed']['values'] == 3216
    assert numeric['v41-uninterrupted']['values'] == 1536
    assert all(numeric[name]['max_abs'] == 0 for name in ['package-default', 'joint-resident', 'joint-all-streamed', 'v41-uninterrupted'])
    # The archived full default log has no captured values. Its earlier
    # training log does: compare every matching label, without treating an
    # empty comparison set as evidence of unchanged drift.
    prior_training = ROOT / 'target/rdna3-streaming/v41-training-regression.log'
    old = logits(prior_training)
    current = logits(evidence / 'v41-default-gate.log')
    assert old and set(old) <= set(current)
    assert all(sorted(current[key]) == sorted(values) for key, values in old.items())
    numeric['archived_training_drift_comparison'] = dict(**binding(prior_training),
        compared_labels=len(old), comparisons=sum(map(len, old.values())),
        identical_reported_drift=True)
    lifecycle_path = evidence / 'cli-lifecycle/qualification.json'
    lifecycle = json.loads(lifecycle_path.read_text())
    assert lifecycle['pass_'] and len(lifecycle['records']) == 6
    assert sha(evidence / 'cli-lifecycle/hierarchos-native-cli.exe') == lifecycle['cli_sha256']
    before = json.loads((evidence / 'before/bindings.json').read_text())
    assert all(sha(evidence / 'before' / Path(name).name) == digest for name, digest in before['sources'].items())
    prior = json.loads((ROOT / 'hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-metadata-upload.json').read_text())
    assert prior['candidate_probe']['sha256'] == sha(evidence / 'before/probe.exe')
    streamed = performance['results']['default']
    assert streamed['decode_submissions']['candidate'] < streamed['decode_submissions']['baseline']
    for row in streamed['runs']['candidate']:
        assert row['measured_payload_reads'] == 0
        assert row['measured_cache_hits'] > 0
        assert row['io']['complete']['cache_peak_bytes'] <= 32 * 1024 * 1024
    source_names = [
        'src/vulkan.rs', 'src/deepseek_v41.rs', 'src/deepseek_v41/model.rs',
        'src/deepseek_v41/backbone.rs', 'src/deepseek_v41/checkpoint.rs',
        'src/deepseek_v41/checkpoint_cache.rs', 'src/deepseek_v41/mixed_linear.rs',
        'src/deepseek_v41/packed_weights.rs', 'src/bin/deepseek_v41_forward_probe.rs',
        'validation/bench_deepseek_v41_expert_upload.py',
        'validation/summarize_deepseek_v41_expert_upload.py']
    sources = {name: sha(ROOT / 'hierarchos-vulkan' / name) for name in source_names}
    base = json.loads((ROOT / 'hierarchos-vulkan/validation/release_parity/deepseek_v41/qualification.json').read_text())
    training = json.loads((ROOT / 'hierarchos-vulkan/validation/release_parity/deepseek_v41/training-qualification.json').read_text())
    frozen = {row['fixture']: row['fixture_sha256'] for row in base['models']}
    frozen.update(training['fixtures'])
    assert all(sha(ROOT / 'hierarchos-vulkan/validation/fixtures' / name) == digest for name, digest in frozen.items())
    result = dict(status='qualified_for_recorded_scope', recorded_utc=datetime.now(timezone.utc).isoformat(),
        device='AMD Radeon Graphics (RDNA3)', absolute_logit_limit=2e-7, rtol=0,
        comparison='strict_less_than', numeric=numeric, tests=tests,
        passed_tests=sum(required.values()), performance=binding(performance_path),
        performance_results=performance['results'], cli_lifecycle=binding(lifecycle_path),
        baseline=binding(evidence / 'before/bindings.json'), baseline_probe=binding(evidence / 'before/probe.exe'),
        candidate_probe=binding(evidence / 'candidate-probe.exe'), sources=sources,
        frozen_cpu_fixtures=frozen,
        rejected_experiments=[
            dict(reason='arena upload staging retained 4 MiB after a streamed call',
                 failure=binding(evidence / 'staging-lifetime-test.log'),
                 resolution='exact-size transient staging; unchanged lifetime assertion passes in the final default suite'),
            dict(reason='8192-range metadata cap churned at 5.5 MiB payload',
                 resolution='32768-range cap, unchanged 32 MiB payload ceiling',
                 limits='unfinished pilot excluded; compilation briefly overlapped its final portion before it was stopped')],
        changes=['record packed payload/scales and LoRA factors in one linear upload batch',
                 'batch each expert triple at load time and join streamed uploads with expert computation',
                 'retire each streamed expert before admitting the next; exact-size transient staging frees at retirement',
                 'share a 32 MiB/32768-range raw payload cache across checkpoint shards, checking size/mtime before hits',
                 'probe records load and checkpoint I/O counters outside timed sections'],
        timing_correction='Old decode_ms_per_token divided a per-token median by --decode a second time. Raw phase timings are used for both legs; previous speedup ratios are unaffected, but old absolute ms/token labels were understated.',
        limits=['scaled 40-layer/384-expert/vocab67 performance fixture; no production throughput claim',
                'performance digest guards the final decoded row; separate CPU gates compare complete logits',
                'performance is the median of three alternating pairs; no Gen9 speed claim',
                'routing readbacks remain; no indirect GPU worklist or new kernel arithmetic',
                'application streaming remains opt-in; no full-base offload or Intel Gen9 performance qualification'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], passed_tests=result['passed_tests'], numeric=numeric), indent=2))


if __name__ == '__main__':
    main()

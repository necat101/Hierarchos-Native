"""Bind source-first loading, CPU/PEFT, application lifecycle and regressions."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--tests', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text())
    cpu = json.loads(args.cpu.read_text())
    cli = json.loads(args.cli.read_text())
    assert memory['status'] == 'passed' and memory['max_abs'] == 0
    assert len(memory['records']) == 23 and len(memory['rejections']) == 3
    assert cpu['status'] == 'qualified' and cpu['profiles_qualified'] == 32
    assert cpu['max_abs'] < 2e-7 and (cpu['atol'], cpu['rtol']) == (2e-7, 0)
    assert set(cpu['source_first_profiles']) == {'llama', 'mixtral', 'qwen2_5_gqa', 'qwen2_5_sliding_tied'}
    assert cli['pass_'] and len(cli['records']) == 6
    cli_binary = args.cli.parent / 'hierarchos-native-cli.exe'
    assert sha(cli_binary) == cli['cli_sha256']
    assert sha(args.memory.parent / 'transformer-logits.exe') == memory['binary_sha256']
    assert all(attempt['binaries']['hierarchos-vulkan-transformer-logits.exe'] == memory['binary_sha256'] for attempt in cpu['attempts'])
    for name, digest in memory['sources'].items():
        assert sha(ROOT / 'hierarchos-vulkan' / name) == digest, name
    for name, digest in cpu['sources'].items():
        assert sha(ROOT / 'hierarchos-vulkan' / name) == digest, name
    regressions = []
    trajectory_values = 0
    for log in args.tests:
        matches = re.findall(r'test result: ok\. (\d+) passed; 0 failed;', log.read_text())
        assert matches and sum(map(int, matches)) > 0, log
        regressions.append(dict(**binding(log), passed=sum(map(int, matches))))
        if log.name == 'v41-uninterrupted-final.log':
            points = re.findall(r'full training .* logits \d+: (\d+) values, worst (\S+)', log.read_text())
            assert sum(int(n) for n, _ in points) == 1536
            assert all(float(drift) == 0 for _, drift in points)
            trajectory_values = 1536
    assert trajectory_values == 1536
    moe = {r['label']: r for r in memory['records'] if r['label'].startswith('mixtral-h512')}
    result = dict(status='qualified_for_recorded_scope', recorded_utc=datetime.now(timezone.utc).isoformat(),
        device='AMD Radeon Graphics (RDNA3)', cpu_logit_atol=2e-7, rtol=0,
        comparison='strict_less_than', cpu_peft_max_abs=cpu['max_abs'],
        source_first_families=['llama', 'mistral', 'qwen2', 'mixtral'],
        application_default='resident', cli_flag='--stream-parameters', gui_default=False,
        cpu_peft=binding(args.cpu), loading_memory_and_formats=binding(args.memory),
        cli_lifecycle=binding(args.cli), regression_logs=regressions,
        cli_binary=binding(cli_binary), summarizer=binding(Path(__file__)),
        fresh_v41_uninterrupted_logits=trajectory_values, fresh_v41_uninterrupted_max_abs=0.0,
        resident_vs_source_first_logits=memory['comparisons'], loader_max_abs=memory['max_abs'],
        mixtral_e8_construction_buffer_peak=dict(resident=moe['mixtral-h512-e8']['resident_loading_peak'],
            source_first=moe['mixtral-h512-e8']['streamed_loading_peak'],
            growth_from_e2=moe['mixtral-h512-e8']['streamed_loading_peak'] - moe['mixtral-h512-e2']['streamed_loading_peak']),
        prior_v41_qualification=binding(ROOT / 'hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-streaming-qualification.json'),
        limits=['other generic families retain their resident initial-loading peak',
                'norms, biases, activations, adapter state and specialized operators remain resident',
                'construction buffer counters exclude driver/pipeline allocations and process RSS',
                'format/memory checks compare resident and streamed execution; independent CPU/PEFT gate is separate',
                'no full-base training offload, production weight run, all-146 coverage, or Intel Gen9 performance claim'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

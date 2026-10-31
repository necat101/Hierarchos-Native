"""Aggregate exact full-stride shader brackets; never treat submissions as model steps."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from rank_kernel_profile import parse, aggregate_shaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    for leg in ['before', 'candidate']:
        profiles[leg] = {}
        for mode in ['resident', 'streamed']:
            path = args.evidence / f'profile-{"" if leg == "before" else "candidate-"}{mode}.log'
            text = path.read_text()
            digests = re.findall(r'v41_probe_logits .*digest=([0-9a-f]+)', text)
            assert digests == ['90e096ebc5bf0659'] * 2
            assert len(re.findall(r'v41_probe_phase phase=prefill ', text)) == 2
            assert len(re.findall(r'v41_probe_phase phase=decode ', text)) == 16
            submissions = parse(path)
            raw = aggregate_shaders(submissions)
            gpu_ms = sum(row['gpu_ms'] for row in submissions)
            dispatches = sum(row['dispatches'] for row in submissions)
            assert sum(row[1] for row in raw.values()) == dispatches
            assert all(row[1] == row[3] for row in raw.values()), 'requires sample stride 1'
            shaders = {name: dict(category=row[0], dispatches=row[1], gpu_ms=row[2],
                                  samples=row[3], share=row[2]/gpu_ms)
                       for name, row in sorted(raw.items(), key=lambda item: -item[1][2])}
            profiles[leg][mode] = dict(log=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_calls=18, passes=2, prefill_tokens=8, decodes_per_pass=8,
                dispatches=dispatches, gpu_ms=gpu_ms, shaders=shaders)
    for mode in ['resident', 'streamed']:
        old, new = (profiles[leg][mode]['shaders'] for leg in ['before', 'candidate'])
        for name in ['transformer_deepseek_v41_moe_gate']:
            assert old[name]['dispatches'] == new[name]['dispatches'] == 720
        for name in ['transformer_deepseek_v41_quantize', 'transformer_deepseek_v41_bf16_cast']:
            assert old[name]['dispatches'] > new[name]['dispatches']
    report = dict(profiles=profiles, scope='same scaled 40-layer/E384 fixture and tokens',
                  sample_stride=1, timing='GPU dispatch brackets summed over 18 complete model calls',
                  limits=['profiling perturbs timings; unprofiled alternating A/B is the latency evidence',
                          'zero-dispatch startup submissions are not model calls',
                          'final-row digest guard is separate from complete-logit CPU gates'])
    (args.evidence / 'profile-aggregate.json').write_text(json.dumps(report, indent=2)+'\n')
    for mode in ['resident', 'streamed']:
        print(mode, {leg: profiles[leg][mode]['gpu_ms'] for leg in profiles}, flush=True)


if __name__ == '__main__':
    main()

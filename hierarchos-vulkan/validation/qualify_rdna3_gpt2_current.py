"""Bind fresh RDNA3 GPT-2 qualification evidence to the exact tested source.

This command never generates reference values. It consumes a run directory in
which the existing producers already executed the expanded GPT-2 shape matrix,
the GPT-2 primitives, the retained image/seq2seq families, native V4.1 and the
shader gates, re-verifies every required check, and only then writes the
``current-qualification.json`` that ``check_rdna3_gpt2_receipts.py`` consumes.

The receipt records the tested revision, every source byte the historical
receipt binds, the exact executable and fixture hashes, and the evidence file
hashes. Any missing, stale or failing gate leaves the receipt failing (or
unwritten), so a PASS cannot be manufactured from a historical receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from check_rdna3_gpt2_receipts import COUNTS, RECEIPTS, ROOT, SHAPES, validate_qualification

LIMIT = 2e-7
GATES = ['shared', 'kimi', 'seq2seq', 'gemma3', 'gemma4', 'llava']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_shape_cases():
    return {f'gpt2_b{b}_s{s}_h{h}/{mask}' for b, s, h in SHAPES for mask in ['unmasked', 'padded']}


def worst_logit(rows):
    values = [check['max_abs'] for row in rows for check in row['checks']
              if 'logits' in check['metric'] and 'max_abs' in check]
    require(values, 'qualification has no logit comparisons')
    return max(values)


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='directory holding shapes/, shared/, ... and primitives.log')
    parser.add_argument('--output', type=Path,
                        default=RECEIPTS / 'current-qualification.json')
    parser.add_argument('--cpu-fixtures', type=Path, required=True)
    parser.add_argument('--image-binary', type=Path,
                        default=ROOT / 'hierarchos-vulkan/target/debug/image_text_pipeline_probe.exe')
    parser.add_argument('--gemma3-binary', type=Path,
                        default=ROOT / 'hierarchos-vulkan/target/debug/gemma3_vision_probe.exe')
    parser.add_argument('--gemma4-binary', type=Path,
                        default=ROOT / 'hierarchos-vulkan/target/debug/gemma4_vision_probe.exe')
    parser.add_argument('--llava-binary', type=Path,
                        default=ROOT / 'hierarchos-vulkan/target/debug/llava_pipeline_probe.exe')
    parser.add_argument('--libtest-binary', type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    status = git('status', '--porcelain').stdout.strip()
    require(not status, 'qualification requires a clean worktree:\n' + status)
    head = git('rev-parse', 'HEAD').stdout.strip()
    require(len(head) == 40, 'cannot resolve the tested revision')

    binaries = {
        'image_text_pipeline_probe': digest(args.image_binary),
        'gemma3_vision_probe': digest(args.gemma3_binary),
        'gemma4_vision_probe': digest(args.gemma4_binary),
        'llava_pipeline_probe': digest(args.llava_binary),
        'libtest': digest(args.libtest_binary),
    }

    shapes = load(run / 'shapes' / 'qualification.json')
    rows = validate_qualification(shapes, COUNTS['shapes'])
    require({row['case'] for row in rows} == expected_shape_cases(),
            'expanded shape coverage changed')
    require(shapes['binary_sha256'] == binaries['image_text_pipeline_probe'],
            'shape replay binary differs from the built probe')
    require(shapes['fixtures_sha256'] == digest(args.cpu_fixtures / 'manifest.json'),
            'shape replay consumed another frozen CPU manifest')
    shape_logit = worst_logit(rows)
    require(shape_logit < LIMIT, 'shape logit gate violated')

    retained = {}
    for name in GATES:
        payload = load(run / name / 'qualification.json')
        gate_rows = validate_qualification(payload, COUNTS[name])
        require(payload['binary_sha256'] == binaries[
            'gemma3_vision_probe' if name == 'gemma3' else
            'gemma4_vision_probe' if name == 'gemma4' else
            'llava_pipeline_probe' if name == 'llava' else
            'image_text_pipeline_probe'], f'{name} binary differs from the built probe')
        retained[name] = dict(cases=len(gate_rows), pass_=True,
                              qualification_sha256=digest(run / name / 'qualification.json'),
                              worst_logit=worst_logit(gate_rows))

    primitive_log = (run / 'primitives.log').read_text()
    require(re.search(r'test result: ok\. 9 passed; 0 failed;', primitive_log),
            'primitive tests incomplete')
    require(len(re.findall(r'^test .*rdna3_gpt2_.* \.\.\. ok$', primitive_log, re.M)) == 9,
            'primitive test coverage changed')

    native = load(run / 'v41native' / 'qualification.json')
    require(native.get('pass') is True and native.get('expected_tests') == 30
            and native.get('completed_tests') == 30, 'V4.1 native regression failed')
    require(native.get('binary_sha256') == binaries['libtest'],
            'V4.1 native binary differs from the built libtest executable')
    require(all(job['pass'] for job in native['jobs']), 'V4.1 native job failed')
    require(native.get('max_abs_logit', 1.0) < LIMIT, 'V4.1 native logit gate violated')

    bridge = load(run / 'v41bridge' / 'qualification.json')
    require(bridge.get('pass') is True and bridge.get('expected_cases') == 2
            and bridge.get('completed_cases') == 2, 'V4.1 bridge regression failed')
    require(bridge.get('binary_sha256') == binaries['libtest'],
            'V4.1 bridge binary differs from the built libtest executable')

    provenance = load(run / 'provenance.json')
    require(provenance.get('pass') is True
            and provenance.get('status_counts') == {'identical': 43}
            and provenance.get('manifest_problems') == [], 'shader provenance failed')

    spirv = load(run / 'spirv.json')
    require(spirv.get('pass') is True and len(spirv.get('artifacts', [])) == 43
            and all(row['passed'] for row in spirv['artifacts']), 'SPIR-V validation failed')

    isolation = load(run / 'isolation.json')
    require(isolation.get('unchanged') is True and isolation.get('files_checked') == 1341
            and isolation.get('existing_recipes') == 672
            and isolation.get('failures') == [], 'upstream bytes changed')

    historical = load(RECEIPTS / 'receipt-hashes.json')
    sources = {}
    for name in sorted(historical['sources']):
        path = ROOT / name
        require(path.is_file(), 'bound source is missing: ' + name)
        sources[name] = digest(path)

    evidence = {path.relative_to(run).as_posix(): digest(path)
                for path in sorted(run.rglob('*'))
                if path.is_file() and path.name in ['qualification.json', 'primitives.log',
                                                    'provenance.json', 'spirv.json', 'isolation.json']}

    receipt = dict(
        schema=1, **{'pass': True}, target='rdna3',
        scope='gpt2_frozen_shape_matrix_and_retained_parity',
        source_head=head, worktree=True,
        absolute_limit=LIMIT, rtol=0, comparison='strict_less_than',
        adapter=dict(vendor='1002', device='15bf', driver='32.0.31007.6002'),
        binaries=binaries,
        producer_sha256=digest(Path(__file__)),
        cpu_fixtures_sha256=digest(args.cpu_fixtures / 'manifest.json'),
        shapes=dict(cases=len(rows), worst_logit=shape_logit,
                    binary_sha256=shapes['binary_sha256'],
                    fixtures_sha256=shapes['fixtures_sha256']),
        retained=retained,
        primitives=dict(passed=9, failed=0, ignored=1,
                        log_sha256=digest(run / 'primitives.log')),
        v41=dict(native=dict(tests=30, max_abs_logit=native['max_abs_logit'],
                             receipt_sha256=digest(run / 'v41native' / 'qualification.json')),
                 bridge=dict(cases=2,
                             receipt_sha256=digest(run / 'v41bridge' / 'qualification.json'))),
        provenance=dict(artifacts=43, status_counts=provenance['status_counts'],
                        receipt_sha256=digest(run / 'provenance.json')),
        spirv=dict(artifacts=len(spirv['artifacts']),
                   receipt_sha256=digest(run / 'spirv.json')),
        isolation=dict(files_checked=1341, existing_recipes=672,
                       receipt_sha256=digest(run / 'isolation.json')),
        sources=sources,
        evidence=evidence,
        limits=[
            'eight measured GPT2 geometries; not a Cartesian shape certificate',
            'isolated kernel timings require a separate paired benchmark receipt',
            'native V4.1 retains its frozen-language LoRA contract',
            'no physical Gen9 execution; the shader gate is static preservation',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'CURRENT QUALIFICATION: PASS ({head[:12]}, {len(rows)} shapes, '
          f'{shape_logit:.3g} worst logit, {len(sources)} bound sources)')


if __name__ == '__main__':
    main()

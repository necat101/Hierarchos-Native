#!/usr/bin/env python3
"""Validate every retained RDNA3 shader artifact with the real ``spirv-val``.

The Phase XV receipt requires a 43-artifact SPIR-V gate.  ``check_shader_
provenance.py`` answers a different question (does each committed artifact
rebuild byte-for-byte), and ``verify_training_shaders.py`` is a whole-manifest
comparison, so neither emits the artifact/sha256/passed/diagnostic records that
``qualify_rdna3_gpt2_current.py`` consumes.  This producer does exactly that and
nothing else.

For every artifact it:

* takes the canonical name and target environment from the sealed Phase XV
  gate, which fixes the expected 43-artifact inventory;
* locates precisely one committed ``.spv`` with that name in the shader tree
  (missing, duplicated or unexpectedly renamed artifacts fail);
* hashes the committed bytes and requires they match the sealed digest, so a
  silently modified shader cannot be validated under the old identity;
* runs the installed ``spirv-val`` with that artifact's target environment and
  records the true exit status and diagnostics;
* marks the artifact ``passed`` only when the validator exits 0, and fails the
  whole gate when any artifact is missing, duplicated, changed or invalid.

Nothing is copied from the sealed receipt but the artifact identities and
target environments; every ``passed`` value is produced by executing the
validator on the bytes in this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / 'hierarchos-vulkan'
SEALED = (CRATE / 'validation' / 'release_parity' / 'multimodal_training' /
          '20261108-rdna3-phase15-qualification' / 'gates' / 'spirv.json')
SHADER_DIRS = [CRATE / 'shaders']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(artifact: str, roots: list[Path]) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        hits.extend(sorted(p for p in root.rglob(artifact) if p.is_file()))
    return hits


def find_validator(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    for name in ('spirv-val.exe', 'spirv-val'):
        found = shutil.which(name)
        if found:
            return Path(found)
    for candidate in sorted(Path('C:/VulkanSDK').glob('*/Bin/spirv-val.exe'), reverse=True):
        return candidate
    raise SystemExit('spirv-val is not installed; the gate cannot be produced')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed', type=Path, default=SEALED,
                        help='sealed Phase XV gate fixing the artifact inventory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validator', type=Path)
    parser.add_argument('--shader-dir', type=Path, action='append')
    args = parser.parse_args()

    sealed = json.loads(args.sealed.read_text(encoding='utf-8'))
    expected = sealed['artifacts']
    require = lambda cond, msg: (_ for _ in ()).throw(ValueError(msg)) if not cond else None
    names = [row['artifact'] for row in expected]
    require(len(names) == 43, f'sealed inventory is {len(names)} artifacts, expected 43')
    require(len(set(names)) == len(names), 'sealed inventory repeats an artifact')

    validator = find_validator(args.validator)
    version = subprocess.run([str(validator), '--version'], capture_output=True, text=True)
    require(version.returncode == 0, 'spirv-val --version failed')
    roots = args.shader_dir or SHADER_DIRS

    records, failures = [], []
    seen: dict[str, list[Path]] = {}
    for row in expected:
        name = row['artifact']
        hits = locate(name, roots)
        seen[name] = hits
        if len(hits) != 1:
            records.append(dict(artifact=name, sha256=None, passed=False,
                                target_env=row['target_env'],
                                diagnostic=f'located {len(hits)} copies'))
            failures.append(f'{name}: located {len(hits)} copies')
            continue
        path = hits[0]
        sha = digest(path)
        if sha != row['sha256']:
            records.append(dict(artifact=name, sha256=sha, passed=False,
                                target_env=row['target_env'],
                                diagnostic='committed bytes differ from the sealed artifact'))
            failures.append(f'{name}: committed bytes changed')
            continue
        run = subprocess.run([str(validator), '--target-env', row['target_env'], str(path)],
                             capture_output=True, text=True)
        diagnostic = (run.stdout + run.stderr).strip()
        passed = run.returncode == 0
        records.append(dict(artifact=name, sha256=sha, passed=passed,
                            target_env=row['target_env'], diagnostic=diagnostic))
        if not passed:
            failures.append(f'{name}: spirv-val exit {run.returncode}')

    report = dict(
        passed=not failures,
        **{'pass': not failures},
        validator=dict(path=str(validator),
                       version=version.stdout.strip().splitlines()[0] if version.stdout.strip() else '',
                       inventory='sealed Phase XV 43-artifact set'),
        artifacts=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'SPIR-V VALIDATION: {"PASS" if not failures else "FAIL"} '
          f'({sum(r["passed"] for r in records)}/{len(records)} artifacts, '
          f'{report["validator"]["version"]})')
    for problem in failures:
        print('  ' + problem)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())

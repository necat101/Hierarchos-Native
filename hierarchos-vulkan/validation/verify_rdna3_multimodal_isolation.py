"""Check the upstream Gen9 shader/fixture bytes and existing provenance entries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='528e3f722bc609cb2d0bdc5dcb032141c662dc5b')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    commit = git('rev-parse', a.baseline).decode().strip()
    paths = git('ls-tree', '-r', '--name-only', commit, '--',
        'hierarchos-vulkan/shaders', 'hierarchos-vulkan/validation/fixtures').decode().splitlines()
    failures = []
    for name in paths:
        path = ROOT / name
        # Git blobs are LF-normalized for text. Compare through Git's normal
        # clean filter so checkout CRLF conversion cannot masquerade as edits.
        expected = git('rev-parse', commit + ':' + name).decode().strip()
        actual = git('hash-object', '--path=' + name, str(path)).decode().strip() if path.is_file() else None
        if actual != expected:
            failures.append(name)
    name = 'hierarchos-vulkan/validation/training_shader_manifest.json'
    old = json.loads(git('show', commit + ':' + name))
    new = json.loads((ROOT / name).read_text())
    # The manifest groups recipes by compiler; flatten recursively to avoid
    # depending on its grouping layout while preserving every old recipe.
    def recipes(value):
        if isinstance(value, dict):
            if 'artifact' in value:
                yield value
            else:
                for v in value.values():
                    yield from recipes(v)
        elif isinstance(value, list):
            for v in value:
                yield from recipes(v)
    existing = {r['artifact']:r for r in recipes(old)}
    current = {r['artifact']:r for r in recipes(new)}
    failures.extend('recipe:' + name for name,r in existing.items() if current.get(name) != r)
    result = dict(baseline=commit, files_checked=len(paths), existing_recipes=len(existing),
        unchanged=not failures, failures=failures,
        limitation='Static preservation check; physical Intel Gen9 execution is not performed on the RDNA3 host.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

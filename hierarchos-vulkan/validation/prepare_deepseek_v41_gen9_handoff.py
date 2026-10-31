"""Package portable source, shaders and immutable CPU gates for Intel Gen9.

No compiler output, virtual environment, remote code or production weights is
needed. Cargo fetches locked registry dependencies on the receiving machine.
Every payload is hashed and the completed archive is read back for verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path,
                        help='qualified receipt; defaults to the latest final-push or expert-upload receipt')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latest = ROOT / 'hierarchos-vulkan/validation/release_parity/deepseek_v41/rdna3-final-push.json'
    qualification_path = args.qualification or (latest if latest.is_file() else latest.with_name('rdna3-expert-upload.json'))
    qualification = json.loads(qualification_path.read_text())
    assert qualification['status'] == 'qualified_for_recorded_scope'
    for name, digest in qualification['sources'].items():
        assert hashlib.sha256((ROOT / 'hierarchos-vulkan' / name).read_bytes()).hexdigest() == digest, name
    files = set()

    def add(path):
        path = path.resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)

    def tree(path):
        for child in path.rglob('*'):
            if child.is_file() and not any(p in {'__pycache__', 'target', '.git'} for p in child.relative_to(path).parts):
                add(child)

    for crate in ['hierarchos-vulkan', 'hierarchos-inference', 'hierarchos-native-cli', 'hierarchos-gui']:
        base = ROOT / crate
        for child in base.iterdir():
            if child.is_file() and child.suffix in {'.toml', '.lock', '.rs', '.c', '.md', '.ps1'}:
                add(child)
        tree(base / 'src')
    vulkan = ROOT / 'hierarchos-vulkan'
    tree(vulkan / 'shaders')
    validation = vulkan / 'validation'
    for child in validation.iterdir():
        if child.is_file() and child.suffix in {'.py', '.json', '.md', '.rs', '.ps1'}:
            add(child)
    tree(validation / 'intel-handoff')
    for child in (validation / 'fixtures').iterdir():
        if child.is_file():
            add(child)
    tree(validation / 'fixtures/deepseek_v41_joint_4141111')
    tree(validation / 'release_parity/deepseek_v41')
    tree(validation / 'release_parity/parameter_streaming')
    final_push = qualification_path.name == 'rdna3-final-push.json'
    evidence = ROOT / ('target/v41-final-push' if final_push else 'target/v41-final-optimizations')
    for name in ['vulkan-regression', 'generic-streaming', 'offload-regression',
                 'v41-default-gate', 'package-default', 'joint-resident',
                 'joint-all-streamed', 'v41-uninterrupted', 'cli-tests',
                 *([] if final_push else ['staging-lifetime-test'])]:
        add(evidence / (name + '.log'))
    if final_push:
        for name in ['paired-preparation.log', 'staging-lifetime.log', 'shader-provenance.json',
                     'push-layouts.json', 'push-layouts.log', 'profile-aggregate.json']:
            add(evidence / name)
        for child in (evidence / 'before').iterdir():
            if child.suffix in {'.rs', '.comp', '.spv', '.json'}:
                add(child)
        for child in evidence.glob('profile-*.log'):
            add(child)
    for child in (evidence / 'ab-release').iterdir():
        if child.is_file() and child.suffix in {'.log', '.out', '.json'}:
            add(child)
    add(evidence / 'cli-lifecycle/qualification.json')
    add(ROOT / 'target/rdna3-streaming/v41-training-regression.log')
    for name in ['PROGRESS_DEEPSEEK_V41_AUDIT.md', 'LICENSE', 'README.md']:
        path = ROOT / name
        if path.is_file():
            add(path)
    tree(ROOT / 'licenses')

    # Also collect literal compile-time includes outside those trees. Resolve
    # relative to the caller, just as Rust does; missing files fail packaging.
    for source in list(files):
        # Before-source snapshots are evidence, not compilation units at their
        # archived location. Their includes resolve in the original crate.
        if source.suffix != '.rs' or source.is_relative_to(ROOT / 'target'):
            continue
        text = source.read_text(encoding='utf-8')
        for relative in re.findall(r'include_(?:str|bytes)!\(\s*"([^"]+)"', text):
            add(source.parent / relative)

    entries = {str(p.relative_to(ROOT)).replace('\\', '/'): p for p in files}
    for source in files:
        if source.name != 'Cargo.toml':
            continue
        def check_dependencies(section):
            if isinstance(section, dict):
                for name, value in section.items():
                    if name == 'path' and isinstance(value, str):
                        assert (source.parent / value / 'Cargo.toml').resolve() in files, (source, value)
                    else:
                        check_dependencies(value)
        check_dependencies(tomllib.loads(source.read_text()))
    manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'bytes': path.stat().st_size} for name, path in sorted(entries.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr('HANDOFF-MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        archive.writestr('START-HERE.txt',
            'Read hierarchos-vulkan/validation/intel-handoff/DEEPSEEK_V41.md.\n'
            'Extract to an ordinary writable folder. Build with cargo -j 1.\n'
            'CPU expectations are frozen; no Python or full model download is needed for the Rust gates.\n'
            'Historical receipts contain original-machine paths and are provenance, not Gen9 qualification.\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, binding in manifest.items():
            raw = archive.read(name)
            assert len(raw) == binding['bytes']
            assert hashlib.sha256(raw).hexdigest() == binding['sha256'], name
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {'archive': args.output.name, 'sha256': digest, 'files': len(manifest),
               'bytes': args.output.stat().st_size, 'uncompressed_bytes': sum(v['bytes'] for v in manifest.values()),
               'archive_readback_verified': True,
               'limits': ['Gen9 must build and pass its own CPU parity gates',
                          'locked registry dependencies require Cargo cache or network',
                          'historical receipts retain original-machine evidence paths']}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

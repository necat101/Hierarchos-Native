"""Verify the batch-4 receipt hashes, source bindings, tracks and evidence closure."""
import hashlib
import json
import re
import zipfile
from pathlib import Path


def find_crate():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'Cargo.toml').is_file() and (parent / 'src/deepseek_v41').is_dir():
            return parent
    raise RuntimeError('could not locate hierarchos-vulkan crate')


CRATE = find_crate()
ROOT = CRATE.parent
D = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(D / 'receipt-hashes.json')
files = {
    path.relative_to(D).as_posix()
    for path in D.rglob('*')
    if path.is_file() and path != D / 'receipt-hashes.json'
}
assert files == set(manifest), 'unlisted or missing receipt files'
assert all(sha(D / path) == digest for path, digest in manifest.items()), \
    'receipt hash mismatch'

sources = read(D / 'source-sha256.json')
assert len(sources) == 1531, 'source binding count changed'
assert all(sha(ROOT / path) == digest for path, digest in sources.items()), \
    'qualified source changed'

full = read(D / 'full-replay-qualification.json')
native = read(D / 'native-avx512-qualification.json')
suite = read(D / 'deepseek-suite-qualification.json')
v41 = read(D / 'native-qualification.json')
assert full['pass'] and full['complete'] and len(full['results']) == 9
assert all(result['pass'] for result in full['results'])
assert native['pass'] and native['complete'] and len(native['results']) == 8
assert all(result['pass'] for result in native['results'])
assert full['unchanged_frozen_files'] == 2748
assert suite['pass'] and suite['verified_passed'] == 217
assert suite['expected_tests'] == 243 and len(suite['ignored']) == 26
assert v41['pass'] and v41['completed_tests'] == 30
assert v41['max_abs_logit'] == 0.0
assert native['retained_cases'] == 95 and native['expanded_shape_cases'] == 16
assert full['absolute_limit'] == 2e-7 and full['rtol'] == 0

test_binary = CRATE / 'target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe'
test_sha = sha(test_binary)
summary = read(D / 'summary.json')
assert summary['pass'] and summary['test_binary_sha256'] == test_sha
assert test_sha == suite['binary_sha256'] == v41['binary_sha256']
assert all(sha(Path(path)) == digest for path, digest in full['binary_sha256'].items())
assert all(full['binary_sha256'][path] == digest
           for path, digest in native['binary_sha256'].items())

assert read(D / 'isolation.json')['unchanged']
assert read(D / 'isolation.json')['files_checked'] == 1341
assert read(D / 'isolation.json')['existing_recipes'] == 672
assert read(D / 'previous-receipt.json')['state_chain_manifest_sha256'] == \
    sha(CRATE / 'validation/release_parity/multimodal_training/20261106-rdna3-v41-state-chain/receipt-hashes.json')
bench = read(D / 'performance-reference.json')
assert sha(ROOT / bench['path'] / 'benchmark.json') == bench['benchmark_sha256']

compressor = (D / 'compressor-regressions.log').read_text()
assert re.search(r'test result: ok\. 2 passed; 0 failed;', compressor)
assert 'batch 2 state chain:' in compressor and 'batch 4 state chain:' in compressor
assert compressor.count('both token VJPs and four state VJPs word-exact') >= 2
assert compressor.count('isolated rows and overwritten slots verified') >= 2
focused = (D / 'focused-state-chain.log').read_text()
assert re.search(r'test result: ok\. 1 passed; 0 failed;', focused)
assert 'batch 4 state chain:' in focused
assert 'both token VJPs and four state VJPs word-exact' in focused
assert 'isolated rows and overwritten slots verified' in focused
assert re.search(r'test result: ok\. 9 passed; 0 failed; 1 ignored;',
                 (D / 'gpt2-primitives.log').read_text())
assert re.search(r'test result: ok\. 2 passed; 0 failed;',
                 (D / 'rdna3-extension-primitives.log').read_text())
assert 'full_replay_exit=0' in (D / 'full-replay.log').read_text()
assert 'native_exit=0' in (D / 'native-avx512.log').read_text()
for marker in ['compressor_exit=0', 'focused_exit=0', 'gpt2_exit=0',
               'extension_exit=0', 'tests_done']:
    assert marker in (D / 'tests-run.log').read_text()

with zipfile.ZipFile(D / 'evidence.zip') as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert len(names) == len(set(names))
    assert len(names) == summary['evidence_files']

for name in ['VALIDATION_STATUS.md', 'RDNA3_MULTIMODAL_PORT.md', 'RDNA3_MULTIMODAL_HANDOFF.md']:
    path = ROOT / name
    text = path.read_text()
    assert '20261107-rdna3-v41-batch4-state-chain' in text, name
    assert not any(line.rstrip() != line for line in text.splitlines()), name
for path in [D / 'README.md'] + [ROOT / name for name in
                                 ['VALIDATION_STATUS.md', 'RDNA3_MULTIMODAL_PORT.md',
                                  'RDNA3_MULTIMODAL_HANDOFF.md']]:
    text = path.read_text()
    assert not any(line.rstrip() != line for line in text.splitlines()), path
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if re.match(r'^[a-zA-Z]+://', target) or target.startswith('#'):
            continue
        target = target.split('#', 1)[0]
        assert (path.parent / target).exists(), (path, target)

print('PASS:', len(manifest), 'receipt files,', len(sources),
      'source bindings, both tracks, batch-4 evidence, CRC and links')

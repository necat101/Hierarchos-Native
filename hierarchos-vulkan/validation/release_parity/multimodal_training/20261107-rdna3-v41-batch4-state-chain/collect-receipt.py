"""Seal the batch-4 state-gradient receipt after both tracks and isolated checks pass."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


def find_crate():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'Cargo.toml').is_file() and (parent / 'src/deepseek_v41').is_dir():
            return parent
    raise RuntimeError('could not locate hierarchos-vulkan crate')


CRATE = find_crate()
ROOT = CRATE.parent
WORK = Path(__file__).resolve().parent
FULL = CRATE / 'target/rdna3-port/state-chain-batch4-full-20261107'
NATIVE = CRATE / 'target/rdna3-port/state-chain-batch4-native-20261107'
FULL_LOG = CRATE / 'target/rdna3-port/state-chain-batch4-full.log'
NATIVE_LOG = CRATE / 'target/rdna3-port/state-chain-batch4-native.log'
DEST = CRATE / 'validation/release_parity/multimodal_training/20261107-rdna3-v41-batch4-state-chain'
PRIOR = CRATE / 'validation/release_parity/multimodal_training/20261106-rdna3-v41-state-chain'
PREV = CRATE / 'validation/release_parity/multimodal_training/20261107-rdna3-v41-batch2-decode-snapshot'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def copy(source, name):
    require(source.is_file(), f'missing evidence: {source}')
    shutil.copyfile(source, DEST / name)


require(not DEST.exists(), f'receipt already exists: {DEST}')
require(WORK.is_dir(), f'missing work directory: {WORK}')

full = read(FULL / 'qualification.json')
native = read(NATIVE / 'qualification.json')
suite = read(FULL / 'deepseek215/qualification.json')
v41 = read(FULL / 'native-v41/qualification.json')
isolation = read(WORK / 'isolation.json')
sources = read(WORK / 'source-sha256.json')
source_changes = read(WORK / 'source-changes.json')

require(full['pass'] and full['complete'] and len(full['results']) == 9
        and all(r['pass'] for r in full['results']), 'full replay incomplete')
require(full['unchanged_frozen_files'] == 2748, 'frozen coverage changed')
require(native['pass'] and native['complete'] and len(native['results']) == 8
        and all(r['pass'] for r in native['results']), 'native AVX512 track incomplete')
require(native['retained_cases'] == 95 and native['expanded_shape_cases'] == 16,
        'native track coverage changed')
require(suite['pass'] and suite['verified_passed'] == 217
        and suite['expected_tests'] == 243 and len(suite['ignored']) == 26,
        'DeepSeek coverage changed')
require(v41['pass'] and v41['completed_tests'] == v41['expected_tests'] == 30
        and v41['max_abs_logit'] == 0.0, 'native V4.1 failed')
require(isolation['unchanged'] and isolation['files_checked'] == 1341
        and isolation['existing_recipes'] == 672 and not isolation['failures'],
        'Gen9 preservation failed')
require(len(sources) == 1531, 'source binding count changed')
require(all(sha(ROOT / path) == digest for path, digest in sources.items()),
        'a source changed during qualification')
require(len(source_changes['refreshed']) == 1, 'unexpected source refresh count')
test_source = 'hierarchos-vulkan/src/deepseek_v41/compressor_tape_tests.rs'
require(test_source in source_changes['refreshed'], 'batch-4 test source is not bound')
require(source_changes['refreshed'][test_source]['current'] == sources[test_source],
        'batch-4 source refresh does not match the source map')

test_binary = CRATE / 'target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe'
test_sha = sha(test_binary)
require(test_sha == suite['binary_sha256'] == v41['binary_sha256'],
        'test binary binding differs')
require(all(sha(Path(path)) == digest for path, digest in full['binary_sha256'].items()),
        'a qualification executable changed')
require(all(full['binary_sha256'][path] == digest for path, digest in native['binary_sha256'].items()),
        'qualification tracks used different binaries')

require('full_replay_exit=0' in FULL_LOG.read_text(), 'full replay did not exit cleanly')
require('native_exit=0' in NATIVE_LOG.read_text(), 'native track did not exit cleanly')
tests_run = (WORK / 'tests-run.log').read_text()
for marker in ['compressor_exit=0', 'focused_exit=0', 'gpt2_exit=0',
               'extension_exit=0', 'tests_done']:
    require(marker in tests_run, f'missing isolated test marker: {marker}')

compressor = (WORK / 'compressor-regressions.log').read_text()
require(re.search(r'test result: ok\. 2 passed; 0 failed;', compressor),
        'compressor regressions incomplete')
require('batch 2 state chain:' in compressor, 'batch-2 state chain output missing')
require('batch 4 state chain:' in compressor, 'batch-4 state chain output missing')
require(compressor.count('both token VJPs and four state VJPs word-exact') >= 2,
        'state-chain word-exact observations missing')
require(compressor.count('isolated rows and overwritten slots verified') >= 2,
        'row-isolation observations missing')
focused = (WORK / 'focused-state-chain.log').read_text()
require(re.search(r'test result: ok\. 1 passed; 0 failed;', focused),
        'focused batch-4 test did not pass')
require('batch 4 state chain:' in focused, 'focused batch-4 state chain output missing')
require('both token VJPs and four state VJPs word-exact' in focused,
        'focused batch-4 word-exact observation missing')
require('isolated rows and overwritten slots verified' in focused,
        'focused batch-4 row-isolation observation missing')
require(re.search(r'test result: ok\. 9 passed; 0 failed; 1 ignored;',
                 (WORK / 'gpt2-primitives.log').read_text()),
        'GPT2 primitive coverage changed')
extension = (WORK / 'rdna3-extension-primitives.log').read_text()
require(re.search(r'test result: ok\. 2 passed; 0 failed;', extension),
        'RDNA3 extension primitives failed')
for name in ['rdna3_builtin_gelu_backward_cpu_words', 'rdna3_siglip_avx2_backward_cpu_words']:
    require(name in extension, f'missing RDNA3 primitive: {name}')

PRIOR.mkdir(exist_ok=True)
require((PRIOR / 'receipt-hashes.json').is_file(), 'prior state-chain receipt is missing')
require((PREV / 'benchmark.json').is_file(), 'prior batch-2 benchmark is missing')

DEST.mkdir(parents=True)

copy(FULL / 'qualification.json', 'full-replay-qualification.json')
copy(FULL / 'deepseek215/qualification.json', 'deepseek-suite-qualification.json')
copy(FULL / 'native-v41/qualification.json', 'native-qualification.json')
copy(NATIVE / 'qualification.json', 'native-avx512-qualification.json')
copy(FULL_LOG, 'full-replay.log')
copy(NATIVE_LOG, 'native-avx512.log')
for name in ['compressor-regressions.log', 'focused-state-chain.log',
             'gpt2-primitives.log', 'rdna3-extension-primitives.log',
             'source-sha256.json', 'source-changes.json', 'isolation.json',
             'tests-run.log']:
    copy(WORK / name, name)
copy(CRATE / 'src/deepseek_v41/compressor_tape_tests.rs', 'compressor_tape_tests.rs')
copy(CRATE / 'validation/qualify_rdna3_native_multimodal.py',
     'qualify_rdna3_native_multimodal.py')
copy(CRATE / 'validation/qualify_rdna3_full_multimodal.py',
     'qualify_rdna3_full_multimodal.py')
copy(WORK / 'verify_isolation.py', 'verify_isolation.py')
copy(WORK / 'verify_receipt.py', 'verify-receipt.py')
copy(Path(__file__), 'collect-receipt.py')

for stage in ['shared77', 'seq2seq', 'gemma4-legacy', 'gemma4-rgb',
              'gemma3', 'llava', 'text166']:
    copy(FULL / stage / 'qualification.json', stage + '-qualification.json')
for stage in ['shared', 'kimi', 'seq2seq', 'gemma3', 'gemma4', 'llava',
              'shapes', 'bridge']:
    copy(NATIVE / stage / 'qualification.json', 'native-' + stage + '-qualification.json')

previous = dict(
    state_chain=PRIOR.relative_to(ROOT).as_posix(),
    state_chain_manifest_sha256=sha(PRIOR / 'receipt-hashes.json'),
    batch2_snapshot=PREV.relative_to(ROOT).as_posix(),
    note='The prior state-chain receipt remains sealed for its original batch-2 binary; '
         'this receipt supersedes it only as the current batch-4 evidence.',
)
write(DEST / 'previous-receipt.json', previous)

benchmark = read(PREV / 'benchmark.json')
require(benchmark['pass'], 'prior benchmark did not pass')
write(DEST / 'performance-reference.json', dict(
    path=PREV.relative_to(ROOT).as_posix(),
    benchmark_sha256=sha(PREV / 'benchmark.json'),
    previous_test_binary_sha256=benchmark['binary_sha256'],
    note='The paired tiny-geometry benchmark is retained for its original binary; '
         'this correctness/coverage extension makes no new timing or speedup claim.',
))

exports = {}
for folder, prefix in [(FULL, 'full-replay'), (NATIVE, 'native-avx512')]:
    for path in folder.rglob('*.safetensors'):
        exports[prefix + '/' + path.relative_to(folder).as_posix()] = sha(path)
write(DEST / 'export-sha256.json', exports)

with zipfile.ZipFile(DEST / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as archive:
    for folder, prefix in [(FULL, 'full-replay'), (NATIVE, 'native-avx512')]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.log', '.txt']:
                archive.write(path, prefix + '/' + path.relative_to(folder).as_posix())
    for name in ['full-replay.log', 'native-avx512.log', 'compressor-regressions.log',
                 'focused-state-chain.log', 'gpt2-primitives.log',
                 'rdna3-extension-primitives.log', 'source-sha256.json',
                 'source-changes.json', 'isolation.json', 'tests-run.log',
                 'compressor_tape_tests.rs', 'qualify_rdna3_native_multimodal.py',
                 'qualify_rdna3_full_multimodal.py', 'verify_isolation.py']:
        path = DEST / name
        archive.write(path, 'isolated/' + name)
with zipfile.ZipFile(DEST / 'evidence.zip') as archive:
    require(archive.testzip() is None, 'evidence ZIP CRC failure')
    names = archive.namelist()
    require(len(names) == len(set(names)), 'duplicate evidence ZIP entries')
    evidence_files = len(names)

summary = dict(
    schema=1,
    **{'pass': True},
    test_binary_sha256=test_sha,
    scope='batch-4 compressor decode-tape state-gradient chain with row-isolation and overwrite-boundary coverage',
    absolute_limit=2e-7,
    rtol=0,
    comparison='strict_less_than',
    full_stages=9,
    frozen_image_cases=93,
    standalone_text_runs=166,
    active_deepseek_tests=217,
    enumerated_deepseek_tests=243,
    ignored_deepseek_tests=26,
    native_v41_tests=30,
    native_v41_max_abs_logit=0.0,
    native_retained_image_cases=95,
    expanded_gpt2_cases=16,
    gpt2_primitive_tests=9,
    rdna3_extension_primitive_tests=2,
    compressor_snapshot_tests=2,
    source_bindings=len(sources),
    gen9_frozen_files=2748,
    protected_gen9_files=1341,
    protected_gen9_recipes=672,
    evidence_files=evidence_files,
    refreshed_source_bindings=len(source_changes['refreshed']),
    previous_state_chain_receipt=PRIOR.relative_to(ROOT).as_posix(),
    performance_reference=PREV.relative_to(ROOT).as_posix(),
    limits=[
        'No physical Intel Gen9 execution',
        'No new performance claim or production-size throughput qualification',
        'No push performed; the receipt and code are ready for the user to publish',
    ],
)
write(DEST / 'summary.json', summary)

readme = f'''# Batch-4 compressor state-gradient chain

The rebuilt test binary `{test_sha}` passes both complete qualification tracks
on the Ryzen Z1 Extreme / Radeon `1002:15bf` host. The strict acceptance gate
remains `max_abs < 2e-7`, `rtol = 0`.

## Added coverage

The existing `deepseek_v41_compressor_decode_tape_snapshots_norm_gain` test now
runs the same snapshot identity at batches 1, 2 and 4. For every batch above 1,
distinct nonzero KV and score state gradients are chained through the completed
position-3 tape and the pending position-2 tape. Both token input gradients must
respond to the terminal state gradient; a dropped state argument fails the test.

Each row is then activated independently with no latent loss. At batch 4 this
checks all four rows: the active row must carry a state VJP, every other row must
remain exactly zero, and neither overwritten pool slot may leak into prefill.
After the live norm gain and live pool cache are overwritten, both token VJPs
and all four state VJPs must remain word-identical, and a restarted forward pass
must show that both token VJPs still respond to the changed gain. The exact
batch-4 observation is:

```
batch 4 state chain: both token VJPs and four state VJPs word-exact after live gain/cache overwrite; isolated rows and overwritten slots verified
```

No oracle, expectation or threshold was changed. The two compressor snapshot
tests remain the same test identities: 243 enumerated DeepSeek tests, 217 active
and 26 original opt-in tests.

## Fresh validation

| Track | Result |
| --- | --- |
| Original frozen replay | All nine stages pass |
| Frozen image pipelines | 93 cases: shared 77, seq2seq 6, Gemma4 2, SigLIP 4, LLaVA 4 |
| Standalone text | 166 resident/streaming runs pass |
| DeepSeek | 217/217 active tests, 243 enumerated, 26 original ignored |
| Native V4.1 | 30/30 executions; logit maximum absolute error 0.0 |
| Separate native AVX512 image track | 95 retained cases, including the two private-AVX2 V4.1 bridge cases |
| Expanded GPT2 | 16 masked/unmasked executions at eight geometries |
| GPT2 primitives | 9 passed, 1 opt-in benchmark ignored |
| RDNA3 GELU/SigLIP primitive extensions | 2 passed |
| Compressor snapshot regressions | 2 passed, including batch 2 and batch 4 state chains |

The two image tracks retain their independent CPU contracts and expectations.
The native V4.1 host process uses AVX512 while its pinned private math scope
continues to use AVX2. Counts above are test executions and overlapping scopes,
not a sum of unique test cases.

## Gen9 and provenance

All 2,748 original frozen files remain unchanged. The frozen manifest is
`{full['frozen_manifest_sha256']}`. The original 1,341 shader/fixture files and
672 shader recipes also match the upstream Gen9 baseline exactly. Physical Intel
execution is outside this RDNA3 run.

`source-sha256.json` binds {len(sources)} source and artifact files. The only
refreshed binding is the batch-4 test source; `source-changes.json` records its
previous and current SHA-256. The replay binds all six executables and verifies
that they stay unchanged. `evidence.zip` retains {evidence_files} raw JSON/log
files, including all bounded DeepSeek shard logs and all per-case comparison
receipts. Large exported SafeTensors remain in the local work directory, with
their hashes in `export-sha256.json`.

The prior [state-chain receipt](../20261106-rdna3-v41-state-chain/README.md)
remains sealed for its original binary and is not rewritten. This receipt is the
current batch-4 evidence; `previous-receipt.json` records the prior manifest hash.

## Performance scope

This correctness extension introduces no new performance measurement or speedup
claim. `performance-reference.json` binds the prior paired streamed/resident
benchmark to its original binary in the
[preceding batch-2 snapshot receipt](../20261107-rdna3-v41-batch2-decode-snapshot/README.md).
Production-size throughput remains unqualified.

## Reproduce

Build serially before starting GPU jobs:

```powershell
cargo test --locked --lib --no-run
cargo build --locked --bin image_text_pipeline_probe --bin gemma4_vision_probe --bin gemma3_vision_probe --bin llava_pipeline_probe --bin hierarchos-vulkan-transformer-logits
python validation/qualify_rdna3_full_multimodal.py --frozen target/rdna3-port/gen9-frozen --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_FROZEN_REPLAY
python validation/qualify_rdna3_native_multimodal.py --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_NATIVE_REPLAY
```

Use fresh output directories and one GPU job at a time. The retained local run is
`target/rdna3-port/state-chain-batch4-20261107/`.
'''
(DEST / 'README.md').write_text(readme, encoding='utf-8', newline='\n')

write(DEST / 'receipt-hashes.json', {
    path.relative_to(DEST).as_posix(): sha(path)
    for path in sorted(DEST.rglob('*'))
    if path.is_file() and path != DEST / 'receipt-hashes.json'
})
print('Collected and sealed', DEST, 'with', evidence_files, 'raw evidence files')

"""Collect this state-chain run only after both qualification tracks pass."""
import difflib,hashlib,json,re,shutil,subprocess,zipfile
from pathlib import Path
CRATE=Path(__file__).resolve().parents[3]
ROOT=CRATE.parent
WORK=Path(__file__).resolve().parent
DEST=CRATE/'validation/release_parity/multimodal_training/20261106-rdna3-v41-state-chain'
OLD=CRATE/'validation/release_parity/multimodal_training/20261106-rdna3-gpt2-shapes'
PREV=CRATE/'validation/release_parity/multimodal_training/20261107-rdna3-v41-batch2-decode-snapshot'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def require(c,m):
    if not c:raise ValueError(m)
def copy(source,name):shutil.copyfile(source,DEST/name)

full=read(WORK/'full-replay/qualification.json')
native=read(WORK/'native-avx512/qualification.json')
suite=read(WORK/'full-replay/deepseek215/qualification.json')
v41=read(WORK/'full-replay/native-v41/qualification.json')
isolation=read(WORK/'isolation.json')
require(full['pass'] and full['complete'] and len(full['results'])==9 and all(r['pass'] for r in full['results']),'full replay incomplete')
require(full['unchanged_frozen_files']==2748,'frozen coverage changed')
require(native['pass'] and native['complete'] and len(native['results'])==8 and all(r['pass'] for r in native['results']),'native AVX512 incomplete')
require(suite['pass'] and suite['verified_passed']==217 and suite['expected_tests']==243 and len(suite['ignored'])==26,'DeepSeek coverage changed')
require(v41['pass'] and v41['completed_tests']==v41['expected_tests']==30 and v41['max_abs_logit']==0.,'native V4.1 failed')
require(isolation['unchanged'] and isolation['files_checked']==1341 and isolation['existing_recipes']==672 and not isolation['failures'],'Gen9 preservation failed')
sources=read(WORK/'source-sha256.json')
require(all(sha(ROOT/p)==v for p,v in sources.items()),'a source changed during qualification')
require(all(sha(Path(p))==v for p,v in full['binary_sha256'].items()),'a binary changed during qualification')
require(all(full['binary_sha256'][p]==v for p,v in native['binary_sha256'].items()),'qualification tracks used different binaries')
test_binary=CRATE/'target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe'
test_sha=sha(test_binary)
require(test_sha==suite['binary_sha256']==v41['binary_sha256'],'test binary binding differs')
for name,count in [('gpt2-primitives.log',9),('compressor-regressions.log',2)]:
    require(re.search(rf'test result: ok\. {count} passed; 0 failed;', (WORK/name).read_text()),name+' incomplete')
require('both token VJPs and four state VJPs word-exact' in (WORK/'compressor-regressions.log').read_text(),'missing state-chain observation')
require('test result: ok. 2 passed; 0 failed;' in (WORK/'rdna3-extension-primitives.log').read_text(),'RDNA3 primitive extensions failed')

# Keep the exact older expanded receipt before refreshing its current-source
# bindings. No oracle, shader, benchmark or historical expectation is changed.
DEST.mkdir(parents=True,exist_ok=False)
shutil.copytree(OLD,DEST/'previous-expanded-receipt')
old_manifest=read(OLD/'receipt-hashes.json')
stale={p:dict(previous=v,current=sha(ROOT/p)) for p,v in old_manifest['sources'].items() if sha(ROOT/p)!=v}
for name in ['shared','kimi','seq2seq','gemma3','gemma4','llava','shapes']:
    shutil.copyfile(WORK/'native-avx512'/name/'qualification.json',OLD/(name+'-qualification.json'))
shutil.copyfile(WORK/'native-avx512/bridge/qualification.json',OLD/'v41-bridge-qualification.json')
shutil.copyfile(WORK/'full-replay/native-v41/qualification.json',OLD/'v41-native-qualification.json')
shutil.copyfile(WORK/'gpt2-primitives.log',OLD/'primitives.log')
shutil.copyfile(WORK/'isolation.json',OLD/'isolation.json')
refresh=dict(schema=1,**{'pass':True},qualification_tracks=['full Gen9 frozen replay','native AVX512 95 cases plus 16 expanded shapes'],
    test_binary_sha256=test_sha,previous_receipt_manifest_sha256=sha(DEST/'previous-expanded-receipt/receipt-hashes.json'),
    previous_receipt_archive=str(DEST.relative_to(ROOT)/'previous-expanded-receipt'),
    refreshed_sources=stale,evidence_receipt=DEST.relative_to(ROOT).as_posix(),
    benchmark_policy='Historical GPU kernel measurement unchanged; no new speedup claimed for this correctness and coverage revision')
write(OLD/'state-chain-refresh.json',refresh)
summary=read(OLD/'summary.json');summary['state_chain_refresh']=refresh
write(OLD/'summary.json',summary)
with (OLD/'README.md').open('a',encoding='utf-8',newline='\n') as f:
    f.write('\nThe [state-chain refresh](../20261106-rdna3-v41-state-chain/README.md) reruns\n'
        'all 95 retained native cases, 16 expanded shapes and nine GPT2 primitive\n'
        'tests against the current source and binaries before refreshing five stale\n'
        'source bindings. It archives the previous receipt bytes. The historical\n'
        'isolated-kernel benchmark is unchanged; this correctness refresh makes no new\n'
        'performance claim. See `state-chain-refresh.json` for the complete bindings.\n')
write(OLD/'receipt-hashes.json',dict(schema=1,
    receipts={p.name:sha(p) for p in sorted(OLD.iterdir()) if p.is_file() and p.name!='receipt-hashes.json'},
    sources={p:sha(ROOT/p) for p in old_manifest['sources']}))
subprocess.run([str(CRATE/'.venv-torch/Scripts/python.exe'),str(CRATE/'validation/check_rdna3_gpt2_receipts.py')],cwd=CRATE,check=True)

for src,name in [
    (WORK/'full-replay/qualification.json','full-replay-qualification.json'),
    (WORK/'full-replay/deepseek215/qualification.json','deepseek-suite-qualification.json'),
    (WORK/'full-replay/native-v41/qualification.json','native-qualification.json'),
    (WORK/'native-avx512/qualification.json','native-avx512-qualification.json'),
    (WORK/'full-replay.log','full-replay.log'),(WORK/'native-avx512.log','native-avx512.log'),
    (WORK/'compressor-regressions.log','compressor-regressions.log'),
    (WORK/'gpt2-primitives.log','gpt2-primitives.log'),
    (WORK/'rdna3-extension-primitives.log','rdna3-extension-primitives.log'),
    (WORK/'source-sha256.json','source-sha256.json'),(WORK/'isolation.json','isolation.json'),
    (Path(__file__),'collect-receipt.py'),
    (CRATE/'src/deepseek_v41/compressor_tape_tests.rs','compressor_tape_tests.rs'),
    (CRATE/'validation/qualify_rdna3_native_multimodal.py','qualify_rdna3_native_multimodal.py'),
]:copy(src,name)
for stage in ['shared77','seq2seq','gemma4-legacy','gemma4-rgb','gemma3','llava','text166']:
    copy(WORK/'full-replay'/stage/'qualification.json',stage+'-qualification.json')
for stage in ['shared','kimi','seq2seq','gemma3','gemma4','llava','shapes','bridge']:
    copy(WORK/'native-avx512'/stage/'qualification.json','native-'+stage+'-qualification.json')
write(DEST/'expanded-receipt-refresh.json',dict(path=OLD.relative_to(ROOT).as_posix(),
    receipt_manifest_sha256=sha(OLD/'receipt-hashes.json'),refreshed_sources=stale))
write(DEST/'performance-reference.json',dict(path=PREV.relative_to(ROOT).as_posix(),
    benchmark_sha256=sha(PREV/'benchmark.json'),previous_test_binary_sha256=read(PREV/'benchmark.json')['binary_sha256'],
    note='Previous paired tiny-geometry timings retained for their original binary; no new timing claim in this correctness and coverage extension'))
prior=WORK.parent/'state-chain-20261106'
after=(CRATE/'src/transformer/vision.rs').read_bytes()
new_gate=b"            // This rounding repair was qualified with the frozen AVX2 image\n            // track. Preserve the native AVX512 graph's established staging.\n            rdna3_tanh_backward: if crate::vendor::rdna3_frozen_avx2_variants(\n                device,\n                aten_vector_width(),\n            ) {"
old_gate=b"            rdna3_tanh_backward: if crate::vendor::rdna3_module_enabled(device) {"
before=after.replace(new_gate,old_gate)
require(before!=after and hashlib.sha256(before).hexdigest()==read(prior/'source-sha256.json')['hierarchos-vulkan/src/transformer/vision.rs'],
    'dispatch patch does not reconstruct the measured failing source')
(DEST/'siglip-dispatch-scope.patch').write_text(''.join(difflib.unified_diff(
    before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),
    fromfile='before/src/transformer/vision.rs',tofile='after/src/transformer/vision.rs')),newline='\n')
copy(prior/'source-sha256.json','prior-attempt-source-sha256.json')
write(DEST/'siglip-dispatch-correction.json',dict(
    failed_prior_attempt=read(prior/'native-avx512/gemma3/qualification.json'),
    primitive_observation=read(prior/'gelu-contract-comparison.json'),
    explanation='The added rounding module was admitted beyond its completed full-graph qualification. Limit it to the frozen AVX2 track, retaining the prior native AVX512 graph selection. Both CPU primitive contracts match the correction; primitive accuracy alone did not qualify the complete AVX512 training graph.',
    targeted_native=read(prior/'siglip-scoped-native-r1/qualification.json'),
    targeted_frozen=read(prior/'siglip-scoped-frozen-r1/qualification.json')))
exports={p.relative_to(WORK).as_posix():sha(p)
    for folder in ['full-replay','native-avx512'] for p in (WORK/folder).rglob('*.safetensors')}
write(DEST/'export-sha256.json',exports)
with zipfile.ZipFile(DEST/'evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for folder in ['full-replay','native-avx512']:
        for p in sorted((WORK/folder).rglob('*')):
            if p.is_file() and p.suffix in ['.json','.log','.txt']:
                z.write(p,p.relative_to(WORK).as_posix())
    for p in [WORK/'probes-build.log',WORK/'tests-build.log',WORK/'focused-state-chain.log']:
        z.write(p,p.name)
with zipfile.ZipFile(DEST/'evidence.zip') as z:
    require(z.testzip() is None,'evidence ZIP CRC failure')
    evidence_files=len(z.namelist())
write(DEST/'summary.json',dict(schema=1,**{'pass':True},test_binary_sha256=test_sha,
    scope='batch-2 completed/pending compressor tape state-gradient chain; restore qualified native AVX512 SigLIP dispatch',
    absolute_limit=2e-7,rtol=0,comparison='strict_less_than',full_stages=9,
    frozen_image_cases=93,standalone_text_runs=166,active_deepseek_tests=217,
    enumerated_deepseek_tests=243,ignored_deepseek_tests=26,native_v41_tests=30,
    native_v41_max_abs_logit=0.,native_retained_image_cases=95,expanded_gpt2_cases=16,
    gpt2_primitive_tests=9,rdna3_extension_primitive_tests=2,compressor_snapshot_tests=2,
    source_bindings=len(sources),gen9_frozen_files=2748,protected_gen9_files=1341,
    protected_gen9_recipes=672,evidence_files=evidence_files,refreshed_expanded_source_bindings=len(stale),
    limits=['No physical Intel Gen9 execution','No new performance claim or production-size throughput qualification','Changes remain uncommitted']))
readme=f'''# Batch-2 compressor state-gradient chain

The rebuilt test binary `{test_sha}` passes both complete qualification tracks
on the Ryzen Z1 Extreme / Radeon `1002:15bf` host. The strict acceptance gate
remains `max_abs < 2e-7`, `rtol = 0`.

## Added coverage

The existing decode snapshot test still covers batches 1 and 2. Batch 2 now
supplies distinct nonzero KV and score state gradients after the completed
position-3 step, then passes the returned state gradients into the pending
position-2 tape's backward call. Both token input gradients must respond to
the terminal state gradient.

Each batch row is also activated independently with no latent loss. Gradients
must remain zero in the other row, and neither overwritten slot may leak a
gradient into the prefill state. After the live norm gain and live pool cache
are overwritten, both token gradients and all four returned state-gradient
buffers must remain word-identical. Replaying forward after the gain overwrite
must change both token gradients, proving sensitivity to the captured gain.

The two existing compressor snapshot tests remain the same test identities:
243 enumerated DeepSeek tests, 217 active and 26 original opt-in tests. This
increment extends the test coverage and qualification tooling. The full native
replay also exposed a SigLIP full-grid regression (`2.1606683731079102e-7`)
caused by admitting the added GELU-backward rounding module to the native
AVX512 graph. Its gate now limits that addition to the frozen AVX2 track and
preserves the established native AVX512 graph selection. Both SigLIP tracks
pass after the correction; expectations and thresholds remain unchanged.
`siglip-dispatch-correction.json` retains the failing run and targeted replays.
`siglip-dispatch-scope.patch` reconstructs the failing source byte for byte
against its recorded SHA-256 and shows the admission-gate correction.

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
| Compressor snapshot regressions | 2 passed, including the extended batch-2 chain |

The two image tracks retain their independent CPU contracts and expectations.
The native V4.1 host process uses AVX512 while its pinned private math scope
continues to use AVX2. Counts above are test executions and overlapping scopes,
not a sum of unique test cases.

## Gen9 and provenance

All 2,748 original frozen files remain unchanged. The frozen manifest is
`{full['frozen_manifest_sha256']}`. The original 1,341 shader/fixture files and
672 shader recipes also match the upstream Gen9 baseline exactly. Physical
Intel execution is outside this RDNA3 run.

`source-sha256.json` binds {len(sources)} source and artifact files. The replay
binds all six executables and verifies that they stay unchanged. `evidence.zip`
retains {evidence_files} raw JSON/log files, including all bounded DeepSeek shard
logs and all per-case comparison receipts. Large exported SafeTensors remain
in the local work directory, with their hashes in `export-sha256.json`.

Five inherited expanded-GPT2 source bindings were stale. All 95 retained native
cases, 16 expanded shapes and nine primitive tests were rerun before refreshing
those bindings and the per-suite receipts. The entire previous expanded receipt
is preserved byte for byte in `previous-expanded-receipt/`;
`expanded-receipt-refresh.json` records the old/new bindings.

## Performance scope

This correctness extension introduces no new performance measurement or speedup claim.
`performance-reference.json` binds the prior streamed/resident ABBA benchmark
to its original binary in the [preceding receipt](../20261107-rdna3-v41-batch2-decode-snapshot/README.md).
That receipt's tiny-geometry measurements remain historical evidence for that
binary. Production-size throughput remains unqualified.

## Reproduce

Build serially before starting GPU jobs:

```powershell
cargo test --locked --lib --no-run
cargo build --locked --bin image_text_pipeline_probe --bin gemma4_vision_probe --bin gemma3_vision_probe --bin llava_pipeline_probe --bin hierarchos-vulkan-transformer-logits
python validation/qualify_rdna3_full_multimodal.py --frozen target/rdna3-port/gen9-frozen --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_FROZEN_REPLAY
python validation/qualify_rdna3_native_multimodal.py --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --output NEW_NATIVE_REPLAY
```

Use fresh output directories and one GPU job at a time. The retained local run
is `target/rdna3-port/state-chain-final-20261106/`. Changes remain uncommitted.
'''
(DEST/'README.md').write_text(readme,encoding='utf-8',newline='\n')
write(DEST/'receipt-hashes.json',{p.relative_to(DEST).as_posix():sha(p)
    for p in sorted(DEST.rglob('*')) if p.is_file() and p!=DEST/'receipt-hashes.json'})
print('Collected and sealed',DEST,'with',evidence_files,'raw evidence files')

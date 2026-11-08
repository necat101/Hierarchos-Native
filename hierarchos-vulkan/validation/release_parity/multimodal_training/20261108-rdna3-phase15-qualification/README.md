# RDNA3 Phase XV frozen GPT-2 and multimodal qualification: 2026-11-08

**Status: RDNA3 numerical qualification PASS.** The frozen GPT-2 producer
executed **16/16** cases on the exact current source, the canonical shared
multimodal producer executed **74/74**, and every retained gate is green.
The nine-stage multimodal producer's first run exited 1 only because one
checkpoint-resume test hit a stale fixed scratch path (`os error 183`); that
test passed in isolation and the complete V4.1 stage reran **30/30**. The
failure and the clean rerun are both preserved below.

Source under test: `a2d68f96802f278fbf79f019d58454270989ade5`. The evidence
base for this receipt is `a26ce0441fe88910dad43085fd68e7edabef78f6`; the
receipt is committed in its child. Production source, shaders, Cargo files and
profiles are unchanged since `a2d68f9`:

```
git diff --name-only a2d68f9..HEAD -- \
  hierarchos-vulkan/src hierarchos-vulkan/shaders \
  hierarchos-vulkan/Cargo.toml hierarchos-vulkan/Cargo.lock
# no output
```

No tolerance, fixture, frozen reference, prior receipt or performance result
was changed. All comparisons use `max_abs < 2e-7`, `rtol = 0`,
`strict_less_than`.

## Fresh GPT-2 frozen shapes

Command:

```
.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_shapes.py run \
  --fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --binary target/debug/image_text_pipeline_probe.exe \
  --output target/phase15/gpt2/shapes
```

| Measure | Result |
| --- | --- |
| Frozen shape cases | **16/16** pass, 0 required failures |
| Logit comparisons | 64, all strict `< 2e-7` |
| Worst logit error | `5.960464477539063e-08` |
| Worst training loss delta | `9.5367431640625e-07` (existing `2e-5` gate) |
| Binary | `d5babaa3ef63f74ea9edc0a993b5b614cf29463e83ab3b41fb0dccf001682436` |
| CPU fixture manifest | `709427881624237c05dccf1e9bdd7593285819304af7d3778b885bc206fbad0c` |
| Producer exit | `0` (`gpt2/run.exit`) |

The sixteen cases are the eight measured geometries (batch/sequence/heads
`2/8/4`, `1/8/4`, `3/8/4`, `2/6/4`, `2/10/4`, `2/16/4`, `2/8/2`, `2/8/8`) in
their unmasked and padded variants. This certifies those geometries, not a
Cartesian product or production-size models.

## Fresh multimodal and retained qualification

| Gate | Result | Worst logit | Worst loss | Binary |
| --- | --- | ---: | ---: | --- |
| Shared CLIP image (canonical) | **74/74** | `1.4901161193847656e-07` | `9.5367431640625e-07` | `d5babaa3…` |
| Shared inventory superset | **77/77** | `1.4901161193847656e-07` | `9.5367431640625e-07` | `d5babaa3…` |
| Kimi K3 shared image | **3/3** | `2.9802322387695312e-08` | `4.76837158203125e-07` | `d5babaa3…` |
| Seq2seq image | **6/6** | `8.940696716308594e-08` | `7.152557373046875e-07` | `d5babaa3…` |
| Native Gemma3 | **4/4** | `1.7881393432617188e-07` | `9.5367431640625e-07` | `a76791a3…` |
| Native Gemma4 | **2/2** | `1.6391277313232422e-07` | `4.76837158203125e-07` | `22141a88…` |
| Native LLaVA | **4/4** | `8.940696716308594e-08` | `4.76837158203125e-07` | `1166db47…` |
| Text resident/streaming | **166/166** | `1.7881393432617188e-07` | — | `76ee7c0a…` |
| Native DeepSeek V4.1 suite | **218/218**, 26 ignored | gate pass | — | `0182c020…` |
| Native V4.1 image training | **30/30** | `0.0` | — | `0182c020…` |
| V4.1 shared-image bridge | **2/2** | `0.0` | — | `0182c020…` |
| GPT-2 primitives | 9 passed, 1 ignored | — | — | `0182c020…` |

The 77-case superset is the full shared fixture inventory; removing the three
`kimi_k3_*` cases gives the canonical 74-case matrix. The three Kimi cases are
also qualified separately. The 44-profile inventory retains its documented
pending encoder-decoder and dedicated runtime-bridge profiles; no profile was
reclassified as green.

DeepSeek V4.1 native reports 30/30 tests, `max_abs_logit = 0.0`, 4 optimizer
updates, 1,360 logit values, 1,866 joint comparisons over 155,444 values with
maximum joint error `1.4901161193847656e-08`. The bridge reports 2/2 cases and
zero logit error.

### Nine-stage producer diagnostic

The full producer command was:

```
.venv-torch/Scripts/python.exe validation/qualify_rdna3_full_multimodal.py \
  --frozen target/rdna3-port/gen9-frozen \
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output target/phase15/multimodal/full
```

Eight stages passed in that run: `shared77`, `seq2seq`, `gemma4-legacy`,
`gemma4-rgb`, `gemma3`, `llava`, `text166`, `deepseek215`. The `native-v41`
stage exited 1 because
`deepseek_v41_native_joint_checkpoint_resume_export_and_freshness` hit
`Cannot create a file when that file already exists. (os error 183)` after
0.41 s. Rerunning the identical test in isolation passed in 6.18 s, and the
complete V4.1 stage then ran **30/30** with `max_abs_logit = 0.0`. The failure
was stale scratch state left by an earlier interrupted run, not a numerical
mismatch. Both `multimodal/stages/native-v41/checkpoint.log` and
`checkpoint-rerun.log` are included; the nine-stage run's exit 1 is recorded in
`multimodal/full-run.exit`.

The two producers required by this phase — GPT-2 shapes and shared multimodal
— both exited 0.

## Static preservation gates

| Gate | Result |
| --- | --- |
| Shader provenance | **43/43** byte-identical rebuilds |
| SPIR-V validation | **43/43** accepted by `spirv-val` |
| Upstream preservation | **1,341** files, **672** recipes unchanged against `528e3f7` |
| GPT-2 primitives | 9 passed, 0 failed, 1 ignored |

No physical Intel Gen9 execution or external shader-oracle run was performed.
The shader gates are static preservation checks, not cross-vendor execution.

## Current-source qualification and guard

`current-qualification.json` was produced by the existing producer from the
fresh run directory:

```
.venv-torch/Scripts/python.exe validation/qualify_rdna3_gpt2_current.py \
  --run target/phase15/current \
  --cpu-fixtures target/rdna3-port/gpt2-shapes-final-cpu \
  --libtest-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe \
  --output validation/release_parity/multimodal_training/\
20261108-rdna3-phase15-qualification/current-qualification.json
```

It re-verifies every gate before writing and binds **134** source bytes,
including `hierarchos-vulkan/src/vulkan.rs`, which the historical source set
omitted. Its `source_head` is `a26ce04…`; the receipt commit is its child.

| Check | Result |
| --- | --- |
| `python validation/check_rdna3_gpt2_receipts.py` | **PASS** (`CURRENT SOURCE QUALIFICATION: PASS`) |
| `python validation/check_rdna3_gpt2_receipts.py --historical-only` | **PASS** |
| `python -m unittest … test_rdna3_gpt2_receipts.py` | **OK**, including the new missing-`vulkan.rs` negative control |

## Performance evidence reuse

No production source, shader, Cargo file or profile changed after `a2d68f9`.
The Phase XIV matched-profile performance receipt therefore remains valid and
was not rerun:

`validation/release_parity/multimodal_training/20261108-rdna3-phase14-qualification/`

| Comparison | T5 speedup | 95% CI | Status |
| --- | ---: | --- | --- |
| `372407b` → final | **1.3687×** | 1.2455–1.4275 | PASS |
| `c7e096d` → final | **1.2588×** | 1.1991–1.3365 | PASS |
| `89af693` → final | **1.0876×** | 1.0717–1.1378 | PASS |
| same-binary session OFF/ON | 0.9783× | 0.9257–1.0345 | INCONCLUSIVE |

Native ViT remains 1.9792× (4×4) and 2.0386× (6×4) versus `372407b`; Llama
1.1862× and GPT-2 1.5150×. The session ON/OFF result remains inconclusive and
is not claimed as an improvement. The Phase XIV receipt remains sealed and
byte-unchanged.

## Artifacts and integrity

| Executable | SHA-256 | Bytes |
| --- | --- | ---: |
| `image_text_pipeline_probe.exe` | `d5babaa3ef63f74ea9edc0a993b5b614cf29463e83ab3b41fb0dccf001682436` | 19,540,480 |
| `gemma3_vision_probe.exe` | `a76791a39755b96f7e34df78315dcc9fb1b1d014444a58058a48f00aeac39d39` | 18,614,784 |
| `gemma4_vision_probe.exe` | `22141a8877f3cfd3e0663fad2bf1dc7a09bc78c0464d3248cdf70f121f41b4b8` | 18,615,808 |
| `llava_pipeline_probe.exe` | `1166db47a8d19ce78173468d52b20a99151dcfaf5178e4d9fbdbcf5d99930471` | 17,987,072 |
| libtest `hierarchos_vulkan-2f708da7cb986ebb.exe` | `0182c02030b83ef2d6ac1b0a1e5723ead8a36dbb4c4a982b577bfa21fa0bc029` | 308,532,224 |
| Phase XIV release `final.exe` | `08b377cb5aa9531fbfbafa7a70ae4f45e309f933fdc7dcd7a05c491dae2d6d61` | 254,941,184 |

The binaries were rebuilt at the qualified source from the same debug profile
used by the producers; the release `final.exe` is the unchanged Phase XIV
performance artifact. Binaries are not byte-reproducible.

`SHA256SUMS` seals every file in this receipt. `verify.py` independently
recomputes the manifest, the 134 source hashes, the case inventories, the
tolerance gates, the known full-producer diagnostic, and the static gates.

## Reproduce

From `hierarchos-vulkan`:

```sh
python validation/check_rdna3_gpt2_receipts.py
python -m unittest discover -s validation -p "test_rdna3_gpt2_receipts.py"
python validation/check_rdna3_multimodal_receipts.py
python validation/release_parity/multimodal_training/\
20261108-rdna3-phase15-qualification/verify.py
```

## Limits

- No physical Intel Gen9 hardware and no external shader oracle; no Gen9 or
  cross-vendor claim. The frozen Gen9 bytes are statically preserved only.
- The nine-stage producer's first `native-v41` attempt failed on stale scratch
  state; the test and the complete stage were rerun clean. The raw failure is
  retained rather than hidden.
- The Python producers replay frozen CPU oracles; CPU oracle generation was not
  rerun. No frozen fixture was regenerated or widened.
- The expanded GPT-2 matrix certifies eight measured geometries, not their
  Cartesian product; audio/video and production-size training remain outside
  this receipt.
- Binaries are not byte-reproducible; evidence is bound by SHA-256, not by
  rebuild identity.

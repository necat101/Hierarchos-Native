# RDNA3 Phase XIV T5 pending-clear qualification: 2026-11-08

**Status: T5 performance PASS. Rust correctness PASS. Full release status
CONDITIONAL, because the Python frozen-oracle GPT-2 16/16 and multimodal
74/74 producers were not rerun in this pass.** The decisive historical T5
comparison is green under profile-matched release builds: final is faster than
both `372407b` and `c7e096d`.

Source under test: `a2d68f96802f278fbf79f019d58454270989ade5`. No tolerance,
fixture, frozen reference, or prior receipt was changed.

## HashMap pending-clear correctness

- Entries still hold `Weak<GpuBufferAllocation>`, never a strong buffer, so
  `DeviceInner -> GpuBufferAllocation -> DeviceInner` cannot return. A dedicated
  test drops 128 zeroed buffers and proves `Arc::strong_count(&device.inner) == 1`
  after pruning.
- Address reuse is structurally excluded while an entry exists: the entry's
  `Weak` keeps the `Arc` control block allocated, so the allocator cannot place
  a new allocation at that key. The new `strong_count() == 0` check in
  `drain_pending_clears` is defense-in-depth: a stale weak is removed instead of
  scheduling a fill for whatever address it names.
- `stale_pending_clear_cannot_fill_an_address_reused_allocation` deterministically
  transplants a dead weak onto a live allocation's key and proves the live
  contents survive and the stale entry is pruned.
- `pending_clears_are_applied_once_per_allocation_without_cross_talk` binds 16
  zeroed targets in one batch and proves each fill is consumed once, by its own
  allocation, and none remain.
- `bounded_pruning_reclaims_abandoned_entries_at_the_threshold` plants
  `PENDING_CLEAR_PRUNE_THRESHOLD + 1` dead entries and proves insertion prunes
  them, leaving one live entry.
- `pending_clear_is_consumed_only_by_its_allocating_thread` proves a non-owner
  bind does not steal the fill and the owner still consumes it.
- `pending_clear_drain_probes_only_bound_buffers` now binds two targets and
  asserts inspections scale with bound buffers, not the 64 unbound live fills.
- Existing readback, first-dispatch, overwrite-cancel, and device-lifetime tests
  remain unchanged and pass.

## Full Rust suite

Exact source `a2d68f9`; debug lib test harness
`hierarchos_vulkan-2f708da7cb986ebb.exe` (`0182c02030b83ef2d6ac1b0a1e5723ead8a36dbb4c4a982b577bfa21fa0bc029`).
Three consecutive default-parallel runs and one serial run:

| Run | Exit | Result | Time |
| --- | ---: | --- | ---: |
| run1 | 0 | ok. 1135 passed; 0 failed; 40 ignored; 0 measured; 0 filtered out; | 79.13 s |
| run2 | 0 | ok. 1135 passed; 0 failed; 40 ignored; 0 measured; 0 filtered out; | 65.28 s |
| run3 | 0 | ok. 1135 passed; 0 failed; 40 ignored; 0 measured; 0 filtered out; | 68.52 s |
| serial | 0 | ok. 1135 passed; 0 failed; 40 ignored; 0 measured; 0 filtered out; | 348.45 s |

Inventory: 1,175 tests; 1,135 passed, 40 ignored, 0 failed. No test was newly
ignored and no assertion was weakened.

## Profile-matched historical performance

All arms are release lib test harnesses built from the exact revision with
`opt-level=3`, `lto="thin"`, `codegen-units=1`, empty `RUSTFLAGS`, default
features, the same fixtures (`target/rdna3-port/gen9-frozen/target/multimodal-gen9`),
batch 1, two warmups, seven measured updates, six ABBA rounds. `372407b` predates
the benchmark's `batch` field; `--allow-legacy-batch1` is recorded and accepted
only for batch 1. No timing, threshold, or fixture changed.

### D. Final same-binary session OFF vs ON

Gate: PASS | batch 1 | legacy batch field False | gate failures none | rounds 6 | measured steps 7

| Case | Baseline median (ms) | Final median (ms) | Speedup | 95% CI | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| native_vit_4x4 | 227.410 | 117.333 | 1.9382 | 1.8838..1.9767 | candidate_faster |
| native_vit_6x4 | 229.500 | 119.939 | 1.9135 | 1.8769..1.9425 | candidate_faster |
| llama | 6.385 | 6.198 | 1.0302 | 0.9761..1.1155 | inconclusive |
| gpt2 | 4.279 | 4.164 | 1.0276 | 0.9997..1.0738 | inconclusive |
| t5 | 10.491 | 10.724 | 0.9783 | 0.9257..1.0345 | inconclusive |

### A. Historical 372407b vs final

Gate: PASS | batch 1 | legacy batch field True | gate failures none | rounds 6 | measured steps 7

| Case | Baseline median (ms) | Final median (ms) | Speedup | 95% CI | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| native_vit_4x4 | 227.623 | 115.005 | 1.9792 | 1.8016..2.0542 | candidate_faster |
| native_vit_6x4 | 238.550 | 117.014 | 2.0386 | 1.8442..2.0698 | candidate_faster |
| llama | 7.673 | 6.469 | 1.1862 | 1.0629..1.3371 | candidate_faster |
| gpt2 | 5.953 | 3.929 | 1.5150 | 1.2108..1.5537 | candidate_faster |
| t5 | 14.602 | 10.668 | 1.3687 | 1.2455..1.4275 | candidate_faster |

### B. Last good c7e096d vs final

Gate: PASS | batch 1 | legacy batch field False | gate failures none | rounds 6 | measured steps 7

| Case | Baseline median (ms) | Final median (ms) | Speedup | 95% CI | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| native_vit_4x4 | 117.723 | 118.393 | 0.9943 | 0.9578..1.0140 | inconclusive |
| native_vit_6x4 | 119.440 | 118.376 | 1.0090 | 0.9492..1.0334 | inconclusive |
| llama | 7.081 | 6.190 | 1.1439 | 1.0696..1.1890 | candidate_faster |
| gpt2 | 4.555 | 3.987 | 1.1426 | 1.0295..1.2381 | candidate_faster |
| t5 | 13.620 | 10.820 | 1.2588 | 1.1991..1.3365 | candidate_faster |

### C. Immediate predecessor 89af693 vs final

Gate: PASS | batch 1 | legacy batch field False | gate failures none | rounds 6 | measured steps 7

| Case | Baseline median (ms) | Final median (ms) | Speedup | 95% CI | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| native_vit_4x4 | 116.835 | 116.840 | 1.0000 | 0.9936..1.0166 | inconclusive |
| native_vit_6x4 | 118.748 | 118.917 | 0.9986 | 0.9851..1.0061 | inconclusive |
| llama | 6.667 | 6.329 | 1.0533 | 0.9670..1.1185 | inconclusive |
| gpt2 | 4.378 | 4.128 | 1.0606 | 1.0045..1.1269 | candidate_faster |
| t5 | 11.748 | 10.803 | 1.0876 | 1.0717..1.1378 | candidate_faster |

The T5 historical regression is eliminated and reversed: final is 1.3687x
faster than `372407b` and 1.2588x faster than `c7e096d`, with intervals entirely
above 1.0. The same-binary session ON/OFF T5 result is 0.9783
[0.9257, 1.0345], inconclusive; it is not claimed as either a gain or a
regression.

## Numerical qualification

Focused groups rerun on the exact debug harness:

| Group | Result |
| --- | --- |
| gpt2 | ok. 9 passed; 0 failed; 1 ignored; 0 measured; 1165 filtered out; |
| deepseek_v41_native | ok. 19 passed; 0 failed; 0 ignored; 0 measured; 1156 filtered out; |
| retained | ok. 6 passed; 0 failed; 0 ignored; 0 measured; 1169 filtered out; |
| kimi | ok. 18 passed; 0 failed; 0 ignored; 0 measured; 1157 filtered out; |
| gemma3 | ok. 3 passed; 0 failed; 0 ignored; 0 measured; 1172 filtered out; |
| gemma4 | ok. 7 passed; 0 failed; 2 ignored; 0 measured; 1166 filtered out; |
| compressor | ok. 18 passed; 0 failed; 0 ignored; 0 measured; 1157 filtered out; |
| seq2seq | ok. 3 passed; 0 failed; 0 ignored; 0 measured; 1172 filtered out; |
| mt5 | ok. 4 passed; 0 failed; 0 ignored; 0 measured; 1171 filtered out; |
| t5_ | ok. 6 passed; 0 failed; 0 ignored; 0 measured; 1169 filtered out; |

- DeepSeek suite: 218/218 active tests pass,
  26 ignored, binary-bound to
  `0182c02030b83ef2d6ac1b0a1e5723ead8a36dbb4c4a982b577bfa21fa0bc029`.
- GPT-2: 9 passed, 1 ignored (the 16/16 frozen shape matrix producer was not
  rerun).
- Retained parity, Kimi, Gemma3, Gemma4, compressor, seq2seq, mT5 and T5 groups
  all pass with zero failures.
- `max_abs < 2e-7`, `rtol = 0`, frozen references untouched.

## Artifacts and provenance

| Role | File | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| final a2d68f9 | `final.exe` | `08b377cb5aa9531fbfbafa7a70ae4f45e309f933fdc7dcd7a05c491dae2d6d61` | 254,941,184 |
| 372407b | `372407b.exe` | `981e1576415c6fc0a4e54979a8a021a462a9d97a70fb3514ca2e9e7bc88bfd51` | 244,469,248 |
| c7e096d | `c7e096d.exe` | `c19707dbe4112c944f57fa47b44a0213b32d0d722ce4173ffefa96ff8ea9a18b` | 254,878,720 |
| 89af693 | `89af693.exe` | `fb93b861fdbbfc4182b985ba6117a40ffdeaadac6bee74db38dfb90958da3b4f` | 254,892,544 |

Rust `1.93.0` (254b59607, LLVM 21.1.8), cargo `1.93.0`,
`x86_64-pc-windows-msvc`. GPU: AMD Radeon Graphics `1002:15bf`, driver
`32.0.31007.6002`, 4,293,918,720 reported bytes. Builds used a shared warm
target directory; each executable was copied to an immutable path immediately
after its build and hashed. Raw rounds, logs, build exits and the binary hashes
are included; `SHA256SUMS` seals the receipt.

## Limits

- No physical Gen9 hardware and no external shader oracle; no Gen9 claim.
- Python frozen-oracle producers (`qualify_rdna3_gpt2_shapes.py` 16/16 and
  `qualify_rdna3_full_multimodal.py` 74/74) were not rerun. Their historical
  receipts remain integrity-verified, not fresh.
- Intermediate revisions `6140771` and `42958d3` were not rebuilt this pass;
  the required A/B/C comparisons and session ablation were.
- Release and test binaries are not byte-reproducible.
- The 89af693 worktree initially had an interrupted checkout; it was restored
  to the exact commit (`git reset --hard 89af693`) before building.

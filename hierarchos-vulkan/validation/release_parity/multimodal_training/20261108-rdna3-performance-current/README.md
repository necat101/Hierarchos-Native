# RDNA3 current-source performance matrix: 2026-11-08

**Status: REGRESSION.** Native ViT gains hold, but T5 is a statistically
significant control regression against the historical baseline. The strict
paired gate therefore fails; this receipt must not be read as a performance
qualification.

Physical target: Ryzen Z1 Extreme, AMD Radeon Graphics `1002:15bf`, driver
`32.0.31007.6002`. Source under test: `3806c8c764b447dce03ce6f2636c3b6cc789b33d`.

Every comparison below is its own six-round ABBA run of the existing
`multimodal_training_throughput_benchmark` test: two warmup updates, seven
measured updates, one GPU job at a time, raw samples and child logs retained.
Ratios are paired round-level medians with a 10,000-draw bootstrap 95% interval.
An interval wholly above 1.0 is `candidate_faster`; wholly below 1.0 is
`candidate_slower`; anything crossing 1.0 is `inconclusive`.

## Exact binaries

| Revision | Role | SHA-256 |
| --- | --- | --- |
| `372407b` | historical baseline | `7e7a713106b76f93425b0b08b0f1ef9cec053935ddfb3323874883c81e29c9ab` |
| `c7e096d` | earlier optimized | `86ad8879ebffb21462e50e1e1c8f41ec7da7f52f127e89bdac44bf0d85e27e3b` |
| `42958d3` | pre-buffer-pool | `6b62b688d7af844ee45e0239415bbb0c30ae571c102f75dfbd7d7d604be097e6` |
| `3806c8c` | current | `cfae56cf9cb63aa207381fa13002bf62470154f654a4548ac0a7263d45bd081d` |

The two later revisions were rebuilt from their exact commits in isolated
worktrees. Each run's `benchmark.json` records the absolute binary paths,
hashes, environment and every raw sample; `hashes.json` binds the run files.

## Results

### Historical `372407b` vs current, batch 1

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native Vit 4x4 | 1.5062 | 1.4867..1.5133 | candidate faster |
| native Vit 6x4 | 1.4483 | 1.4301..1.5111 | candidate faster |
| Llama | 0.9572 | 0.9463..1.0252 | inconclusive |
| GPT-2 | 1.0722 | 1.0462..1.2221 | candidate faster |
| T5 | **0.9453** | **0.9320..0.9538** | **candidate slower — gate failure** |

The GPT-2 cross-arm signatures differ by the documented one-to-two ULP, so
this run uses `--allow-cross-arm-signature-difference`; within-arm words are
exact. Llama's interval crosses 1.0 and is not evidence of a regression.

### Historical bracket `372407b` vs `c7e096d`, batch 1

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native Vit 4x4 | 1.5103 | 1.3466..1.5202 | candidate faster |
| native Vit 6x4 | 1.4840 | 1.4721..1.5610 | candidate faster |
| Llama | 1.0040 | 0.9939..1.1511 | inconclusive |
| GPT-2 | 1.1151 | 1.1070..1.1425 | candidate faster |
| T5 | 0.9978 | 0.9897..1.0020 | inconclusive |

T5 is neutral at `c7e096d`, so the regression was not present in the earlier
optimized tree.

### Pre-buffer-pool `42958d3` vs current

| Case | Batch 1 | Batch 2 | Batch 4 |
| --- | ---: | ---: | ---: |
| native Vit 4x4 | 1.0086 (1.0002..1.1186) | 1.0086 (0.9955..1.0207) | 1.0174 (0.9648..1.0235) |
| native Vit 6x4 | 1.0064 (1.0020..1.0073) | 1.0026 (0.9921..1.0068) | 1.0504 (0.9608..1.0716) |
| Llama | 1.0111 (0.9955..1.0407) | 1.0171 (0.9807..1.1320) | 1.0149 (0.9086..1.0280) |
| GPT-2 | 1.0042 (0.9894..1.0204) | 0.9994 (0.9366..1.0787) | 1.0519 (0.8371..1.0956) |
| T5 | 1.0002 (0.9840..1.0100) | 1.0046 (0.9052..1.0078) | 1.0030 (0.8884..1.0552) |

All three runs are word-exact and pass the control gate. At batch 1, T5 is
`1.0002`, so the lifetime-fix lineage and the pool add/revert added no
measurable T5 cost relative to `42958d3`.

### Session OFF vs ON on current, batch 1

| Case | Speedup | 95% interval | Conclusion |
| --- | ---: | --- | --- |
| native Vit 4x4 | 1.4993 | 1.4910..1.6510 | candidate faster |
| native Vit 6x4 | 1.4556 | 1.4491..1.4675 | candidate faster |
| Llama | 1.0050 | 0.9902..1.0531 | inconclusive |
| GPT-2 | 0.9871 | 0.9634..1.0027 | inconclusive |
| T5 | **0.9869** | **0.9374..0.9982** | **candidate slower — gate failure** |

Session mode preserves the native ViT gains but costs T5 about 1.3% in this
run. The interval is narrow enough to fail the strict control gate; it is a
marginal result and a candidate for a repeat on a quieter host.

## Interpretation

- The `1.54–1.56x` historical native ViT claim is reproduced and remains valid:
  `1.506x` / `1.448x` at batch 1 against `372407b`.
- The T5 regression against `372407b` is real on this host and is **not** caused
  by the final lifetime fix or the reverted buffer pool: both historical
  brackets bracket it. It entered between `c7e096d` and `42958d3`, a four-commit
  window that includes the deferred zero-fill lifetime repair and pending-clear
  changes. It has not been bisected to one commit.
- Current and `42958d3` are neutral across batches 1, 2 and 4, so the current
  tree did not add measurable throughput cost on top of the pre-pool tree.
- Batch-2 and batch-4 comparisons are word-exact but inconclusive; they do not
  support a batch-scaling speedup claim against `42958d3`.
- The session T5 cost is separate and marginal; it does not explain the larger
  historical T5 gap.

## Limits

No production-size model throughput is claimed. These are frozen tiny training
geometries. The host was on its normal power profile; no fan or clock control
was applied. The regression window is localized, not bisected. All source,
fixture and tolerance data remain unchanged by this measurement.

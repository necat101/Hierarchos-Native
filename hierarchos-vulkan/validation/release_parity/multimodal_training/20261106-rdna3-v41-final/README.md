# Native V4.1 submission, upload and routing optimization

## Summary

Native V4.1 training now records model construction, the retained forward graph
and the reverse graph in one compute session. Component upload flushes publish
into that session, and forward routing counts are carried through the retained
tape for both the reverse recompute and the adapter-gradient filter. Arithmetic,
kernel selection, dispatch order and buffer contents are unchanged.

Same-binary ABBA, four rounds, seven measured updates, two warmups, software FMA
in both arms:

| Case | Baseline, ms | Session, ms | Ratio | Submissions/step |
| --- | ---: | ---: | ---: | ---: |
| `native_vit_4x4` | 377.8380 | 291.8544 | 1.295x | 970 -> 509 |
| `native_vit_6x4` | 534.4946 | 439.3253 | 1.217x | 970 -> 509 |
| `llama` control | 17.3721 | 17.0755 | 1.017x | 5 -> 5 |
| `gpt2` control | 14.1738 | 13.8799 | 1.021x | 5 -> 5 |
| `t5` control | 39.2459 | 39.2352 | 1.000x | 7 -> 7 |

Every final logit word, image digest and adapter digest matched across all eight
runs. The shared controls keep their original submission counts. The baseline
arm is the same binary with `HIERARCHOS_VULKAN_LAYER_SESSION=0`.

The profile records routing-count readbacks falling from 100 to 64 per step,
while the direct queue-submission count falls from 970 to 509. The remaining
funnels are host uploads, norm-weight duplication and scalar readbacks; the
largest are listed in `profile.log` and are the next measured targets.

## Full nine-stage replay

The same binary passed all nine published stages:

- `shared77`, `seq2seq`, `gemma4-legacy`, `gemma4-rgb`, `gemma3`, `llava`,
  `text166`, `deepseek215`, `native-v41`
- `pass: true`, `complete: true`
- 2,748 frozen files unchanged
- frozen manifest SHA-256:
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- strict gate unchanged: `max_abs < 2e-7`, `rtol = 0`

The native 30-test qualifier also passes with 1,360 logit values,
`max_abs_logit = 0.0`, joint `max_abs = 1.4901161193847656e-08`, and
checkpoint/resume/export verified.

## Source changes

- `model_train.rs`: one outer `begin_compute_session` around model construction,
  forward, backward and head-gradient staging.
- `upload.rs`: a component flush inside an active session publishes into that
  session instead of calling `submit_and_wait`; outside a session it still
  waits.
- `vulkan.rs`: session ownership query used by the upload flush.
- `backbone.rs`, `backbone_tape.rs`, `mixed_moe_grad.rs`: forward returns the
  gate's live-row counts; the retained tape reuses them in reverse and at the
  adapter-gradient boundary.
- Earlier reverse-graph session edits in `attention_grad.rs`,
  `attention_projection_grad.rs`, `attention_tape.rs`, `compressor_grad.rs`,
  `engram_grad.rs`, `linear_grad.rs` and `backbone_tape.rs` remain in the same
  submission boundary.

## Reproduce

```powershell
python validation/qualify_deepseek_v41_native_pipeline.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --output target/rdna3-port/full-multimodal-20261106/v41-routing2-qualifier-r1 `
  --target rdna3

python validation/benchmark_rdna3_v41_session.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-routing2-ab-r1 `
  --rounds 4 --steps 7

python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261106/final-replay-r2 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe
```

## Files and limits

- `benchmark.json`: four-round ABBA medians and submission counts.
- `native-qualification.json`, `native-run-status.json`, `native-qualifier.log`:
  the 30-test native gate.
- `full-replay-qualification.json`, `full-replay.log`: the joined nine-stage
  replay and its pass/fail rows.
- `profile.log`: session-on submission profile.
- `benchmark.log`: runner output.
- `receipt-hashes.json`: SHA-256 bindings for this receipt.

The measurements cover the frozen tiny training geometries and published
replay matrix. They do not certify production-size throughput. The full replay
uses the fixed test binary and did not build the model again.

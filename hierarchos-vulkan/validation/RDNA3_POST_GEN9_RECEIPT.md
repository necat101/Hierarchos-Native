# Physical RDNA3 audit of the final Gen9 port: 2026-11-02

The `2d18ac7` handoff passes the recorded DeepSeek V4.1 Flash and previously
qualified parameter/MoE streaming scopes on the Ryzen Z1 Extreme / AMD Radeon
Graphics (`1002:15bf`, driver `32.0.31007.6002`). No introduced regression was
found in those scopes. Streaming remains opt-in through `--stream-parameters`
or the unchecked GUI toggle.

The expanded RDNA3 forward matrix is **not fully qualified**: 164/166 runs
pass. Padded GPT-2 wide exceeds the strict gate in both storage modes. The
preserved pre-Gen9 executable reproduces every failing candidate logit word,
so this is an existing expanded-matrix parity gap, not an introduced regression.
It remains open; no tolerance, CPU expectation or model arithmetic was changed
to obtain a pass. Diagnostic arithmetic experiments were discarded.

The [receipt](release_parity/rdna3_post_gen9/20261102/qualification.json) and
[evidence archive](release_parity/rdna3_post_gen9/20261102/evidence.zip) retain
the passing and failing reports, hashes, native logs and paired measurements.
The original Gen9 archives and original handoff manifest remain unchanged.

## Correctness

| Gate | Physical RDNA3 result |
| --- | --- |
| Selected native suite | 269 passed: 190 V4.1, 38 Vulkan, seven streaming, 17 offload, 12 cache, one Qwen3-Next backward and four explicit V4.1 gates |
| V4.1 default logit gate | 201,714 values, worst `1.4901161193847656e-7` |
| V4.1 package/resident/streamed/PEFT trajectory | 11,184 logits bit-exact against frozen CPU expectations |
| Previous 32 RDNA streaming profiles | 122/122 forward runs; worst `8.940696716308594e-8`; resident/streamed words identical |
| Expanded forward matrix | 164/166 runs across 43 profiles; not fully qualified |
| Cached/full-prefix generation | 72/72 requests, worst `8.940696716308594e-8`; tokens and storage words match |
| Additional native GPT-2 checks | 48 passed, including generation/cache and adapter persistence checks; some overlap the cache suite above |
| CLI | 41 passed |
| Shaders and ABI | 569 active artifacts reproduce; 429 push blocks / 108 host structs agree |
| Workspace | Four crates / 65 binaries, all-target check passes |

Every accepted logit comparison retains `max_abs < 2e-7`, `rtol=0`.
The inference/training fixtures are the retained scalable checkpoints, not
production weights or a CUDA run. Forward CPU coverage excludes the existing
padded-query rows; storage identity includes every row. SmolLM3's cache request
still uses its documented full-prefix fallback. This does not qualify all
146 registry identifiers or every architecture's CPU PEFT training.

## Existing expanded-matrix gap

`gpt2_wide/padded` has two failing observable values per storage mode. Its
worst drift is `2.384185791015625e-7`; resident and streamed outputs match
exactly. The old benchmark executable has SHA-256
`2ff8332e4adba9db238286c630143e54c4f4dfcf11241acb3a6dd5de848c680b`, already
bound by [the pre-Gen9 submission receipt](release_parity/deepseek_v41/rdna3-submission-batching.json).
It produces the same 3,840 logit words as the candidate, including all padding
and both failing values. The immutable CPU package and expectations come from
the [final Gen9 archive](release_parity/gen9_final/20261101/evidence.zip).

The runbook's original fail-fast runner stopped at this expanded-matrix
failure before exercising V4.1. This audit continued the native and generation
gates separately, keeping the failed forward result. The runner now offers
`--keep-going` to collect all numerical/test failures in one pass. It still
returns nonzero for any failing scope, rejects empty test filters and refuses
to continue without establishing the requested physical AMD adapter. Seven
positive/negative controls cover these rules and run in CI.

## Performance observations

Three alternating pairs compare the preserved qualified RDNA3 handoff probe
with the final Gen9-port probe. Both execute on the same AMD device, after
builds/tests finish, with two measured steps and one warmup per invocation.
Every measured and guard digest matches the frozen value.

| V4.1 scaled checkpoint | Previous handoff | Gen9 port on RDNA3 | Median latency reduction |
| --- | ---: | ---: | ---: |
| Resident prefill8, ms | 518.6605 | 391.8960 | 24.4% |
| Resident decode, ms/token | 307.7980 | 153.1755 | 50.2% |
| Streamed prefill8, ms | 1286.9510 | 1065.1240 | 17.2% |
| Streamed decode, ms/token | 511.7480 | 417.4815 | 18.4% |

Eight additional architecture controls each use three alternating rounds,
five warmups and 31 measured forwards. Every invocation passes its CPU gate,
and every baseline/candidate/storage word matches. Streamed median latency is
3.1–57.8% lower across GPT-2 wide, Llama wide, Qwen3-Next, Qwen3.5 mixed,
Qwen4-Exp mixed PLE, MiniMax-M3, GPT-OSS and Mixtral. GPT-2 wide's timing uses
the passing unmasked case; it does not close the padded case above. The older
generic benchmark supports streaming only, so this experiment does not claim
a paired generic resident improvement.

These are observations on small retained checkpoints. Timing varies with host
and driver load; no production throughput or Gen9 timing is inferred here.

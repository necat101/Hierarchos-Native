# RDNA3 full Gen9-green multimodal replay, 2026-11-06

Physical target: Ryzen Z1 Extreme, CPUID `00a70f41`, AMD Radeon Graphics
`1002:15bf`, driver `26.10.07.06 (LLPC)`. This is a worktree qualification;
no new commit or publication is implied.

This receipt replays every stage of the published green Gen9 multimodal matrix
on the physical RDNA3 adapter against the read-only frozen AVX2 oracles. The
acceptance policy is unchanged: `max_abs < 2e-7`, `rtol = 0`, strict
comparison.

## Matrix result

| Stage | Coverage | Result |
| --- | ---: | --- |
| `shared77` | 77 shared-image cases | PASS |
| `seq2seq` | 6 encoder-decoder cases | PASS |
| `gemma4-legacy` | 1 native Gemma4 case | PASS |
| `gemma4-rgb` | 1 native Gemma4 RGB case | PASS |
| `gemma3` | 4 SigLIP/Gemma3 cases | PASS |
| `llava` | 4 native LLaVA cases | PASS |
| `text166` | 166 resident/streaming text runs | PASS |
| `deepseek215` | 215 active DeepSeek tests | PASS |
| `native-v41` | 30 native V4.1 tests | PASS |

The replay covers 93 image cases (77 shared, 6 seq2seq, 2 Gemma4, 4 Gemma3,
4 LLaVA), 166 text runs, 215 active DeepSeek tests with the original 26 opt-in
tests ignored, and all 30 native V4.1 tests. `qualification.json` records
`pass: true`, `complete: true`, and the exact stage commands.

The 2,748-file frozen Gen9 manifest is unchanged. No physical Gen9 execution
is claimed; Gen9 shaders, fixtures and recipes were not modified.

## Frozen inputs and binaries

- Frozen manifest SHA-256:
  `a9e519008e712105ceb1b135227398785617db3eb03c65e7518bb7f225966636`
- Published baseline SHA-256:
  `e20ab6b495cf1407584dcfed2778f83de1abc5518a249c049089bf60d51564f9`
- Test binary SHA-256:
  `46ad23463d1d788eadfb4b9d40ae80029dda7d016f8311cd3a47ab1cb3c0765c`
- Image probe SHA-256:
  `70d8505dd7ac678f5437cd9873919e6f0063ae542d12cb0133ca9fd0b22164dd`
- Gemma3 probe SHA-256:
  `d0a94728f9676ed57d85fa7c4bfed73be67249aae87dc0c5df7e07c08b5bd16c`
- Gemma4 probe SHA-256:
  `f5d5bad8daa05284b5beb9e7e8da0c4e97e5eb142f0e14939056743c1eba5154`
- LLaVA probe SHA-256:
  `f923e19a5afeda5c094d18a0bec9e52b716b0e9fc51f3bd6d9e9879fe5d91746`
- Transformer logits probe SHA-256:
  `4830c9db5be9fbb158dc5a9cebdc1297196900b4214955dca259dfd2dc5b51b8`

## V4.1 optimization measurement

The native V4.1 hardware-FMA candidate is selected only on the qualified
RDNA3/Zen4 profile and was tested in the same binary with grouped uploads in
both arms. Four ABBA rounds, seven measured updates and two warmups produced
identical final logits, image digests and adapter digests:

| Case | Software, ms | Hardware FMA, ms | Ratio |
| --- | ---: | ---: | ---: |
| `native_vit_4x4` | 328.0540 | 329.3742 | 0.996x |
| `native_vit_6x4` | 482.5545 | 482.5499 | 1.000x |
| `gpt2` control | 13.7083 | 13.7463 | 0.997x |
| `llama` control | 16.4960 | 16.5759 | 0.995x |
| `t5` control | 37.9021 | 37.8492 | 1.001x |

The candidate is correctness-preserving but performance-neutral on these two
native geometries; the earlier two-round `1.047x` observation on `4x4` did not
replicate, so no speedup is claimed and it is not a basis for changing the
numerical path. The previously qualified GPT-2 backward LDS path remains the
measured optimization in this worktree (5.26x and 10.60x on the two reported
kernel geometries). None of these isolated measurements is a production-size
training throughput claim.

## Reproduce

Run from `hierarchos-vulkan` with the frozen root and the final test binary:

```powershell
python validation/qualify_rdna3_full_multimodal.py `
  --frozen target/rdna3-port/gen9-frozen `
  --output target/rdna3-port/full-multimodal-20261106/final-replay-r1 `
  --test-binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe
```

The benchmark is:

```powershell
python validation/benchmark_rdna3_v41.py `
  --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe `
  --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9 `
  --output target/rdna3-port/full-multimodal-20261106/v41-fma-benchmark-r4 `
  --rounds 4 --steps 7
```

## Files and limits

- `qualification.json`: the joined nine-stage replay result.
- `summary.json`: counts, stage status, binary and benchmark bindings.
- `v41-fma-benchmark.json`: the four-round ABBA result.
- `replay.log` and `v41-fma-benchmark.log`: captured runner output.
- `receipt-hashes.json`: SHA-256 bindings for the receipt files.

The receipt is limited to the frozen tiny training geometries and the published
matrix. It does not certify production-size throughput, audio/video coverage,
or a physical Gen9 run.

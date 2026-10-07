# Rejected vendor-kernel candidates

These modules were written for Intel Gen9 (HD Graphics 520-class) and measured on
the target hardware. None of them is selected by `src/vendor.rs`; they are kept
so a future Intel generation (Xe/Arc) or driver can re-run the same A/B without
rewriting the kernels.

| Module | Family | Measured on HD Graphics 520 | Why it is not admitted |
| --- | --- | --- | --- |
| `silu_forward_intel_gen9` | `silu-forward` | 0.78x of the portable module | Slower: 64 invocations x 4 elements made the elementwise pass slower, not faster. |
| `silu_backward_intel_gen9` | `silu-backward` | 0.85x of the portable module | Slower for the same reason. |
| `adamw_intel_gen9` | `adamw` | within noise on timing, **1-ulp mismatch** at element 64 | The different loop shape changed how the driver contracted the update expression, so the optimizer was no longer bit-identical. Bit-exactness is a hard requirement for every variant. |
| `parameter_matmul_forward_intel_gen9` | `parameter-matmul-forward` | 0.79x at 8x448x512, 0.98x at 6x96x320, 1.12x at 5x64x192 | The transposed-weight (`weight[in_col * output_dim + out_col]`) portable module is already fully coalesced - 0.193 ms at 8x448x512 against 0.771 ms for the row-major kernel at the same shape - so workgroup staging and two barriers per k-slice cost more than the loads they replace. Bit-exact (`max_ulp_delta=0`) but slower. |
| `parameter_matmul_bias_forward_intel_gen9` | `parameter-matmul-bias-forward` | 0.80x at 8x448x512, 1.00x at 6x96x320, 1.12x at 5x64x192 | Same addressing and same verdict as the unbiased module. |

A fourth experiment (a short-k direct path inside the tiled matmul module) is not
kept here: it measured 0.70x at k=8 because the workgroup-memory declarations in
the module cost occupancy for *all* shapes, regardless of the branch taken. The
admitted design instead gates on the host with `INTEL_TILED_MIN_INPUT_DIM`.

Run any of them through the committed A/B harness:

```powershell
$env:HIERARCHOS_VULKAN_VENDOR_BENCH='silu-forward'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_A='shaders/silu_forward.spv'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_B='shaders/vendor_experiments/silu_forward_intel_gen9.spv'
$env:HIERARCHOS_VULKAN_VENDOR_BENCH_SHAPE='len=16384'
cargo test --release --lib candidate_module_experiment_from_env -- --nocapture --test-threads=1
```

The matmul hook takes `SHAPE=rowsxinput_dimxoutput_dim`. The two
parameter-matmul candidates were driven through the `linear-forward` slot (the
same three-binding `(rows, input_dim, output_dim)` contract, so the A/B is
apples-to-apples) with `SHAPE=8x448x512`, and the admitted families print their
own A/B table from
`cargo test --release --lib intel_variants_are_bit_exact_with_portable_kernels --
--nocapture --test-threads=1`.

The harness prints the portable/variant ratio and fails if the two modules are
not bit-identical, so a future candidate can only be admitted by winning on time
without losing exactness.

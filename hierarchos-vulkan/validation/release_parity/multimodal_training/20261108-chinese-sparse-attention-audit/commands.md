# Reproduction commands

All commands ran from `hierarchos-vulkan` on the documented RDNA3 host.

```powershell
# Build and enumerate the current-source libtest binary.
cargo test --lib --no-run

# Targeted native filters (serial GPU; logs under rust/).
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe qwen --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe kimi --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe minimax --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe "deepseek_v41::attention" --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe "deepseek_v41::attention_grad" --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe "deepseek_v41::indexer" --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe "deepseek_v41::compressor" --test-threads=1 --nocapture
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe "deepseek_v41::mixed" --test-threads=1 --nocapture

# Frozen Chinese image-training subset (corrected attempt 2).
python validation/qualify_shared_image_pipeline.py run   --fixtures target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3   --binary target/debug/image_text_pipeline_probe.exe   --output target/chinese-audit-20261108/shared   --target rdna3 --device-id 0x15bf   --profiles qwen2 qwen2_5_gqa qwen2_5_sliding_tied qwen3_5_full qwen3_5_linear   qwen3_5_mixed qwen3_5_moe qwen4_exp_linear qwen4_exp_qsa qwen4_exp_ple   qwen4_exp_mixed_ple qwen3_next deepseek_v4 kimi_k25_text kimi_k3_kda   kimi_k3_mla kimi_k3_text minimax_m2 minimax_m3_dense minimax_m3

# Complete DeepSeek suite.
python validation/qualify_rdna3_deepseek_suite.py   --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe   --output target/chinese-audit-20261108/deepseek218 --shard-size 8 --host-width 16

# Native V4.1 pipeline.
python validation/qualify_deepseek_v41_native_pipeline.py   --binary target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe   --output target/chinese-audit-20261108/native-v41 --target rdna3 --device-id 0x15bf

# Complete default-parallel Rust suite.
target/debug/deps/hierarchos_vulkan-2f708da7cb986ebb.exe --test-threads=16

# Receipt verification.
python validation/release_parity/multimodal_training/20261108-chinese-sparse-attention-audit/verify.py
```

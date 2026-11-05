# DeepSeek V4.1 native vision component gate (Gen9)

Four tests pass on Intel HD Graphics 520. Two unchanged released ViT/aligner
CPU fixtures (4x4 and 6x4 RGB, including spatial padding) compare 200 metrics
through two AdamW updates: all forward features, projected features and updated
parameters are bit-exact; the largest parameter/input gradient difference is
1.862645149230957e-9. Every compared value is strictly below 2e-7, rtol=0.

Independent controls cover eight SDPA cases (2560 output/adjoint words), sixteen
aligner Linear cases (1920 words), and eight strided RMS cases (640 words).
The separate HC repeat controls cover six cases (1664 words).

The reference is deepseek-ai/DeepSeek-V4.1-Flash-1017 at revision
1bae0e1088993b9b82de08f59de5be5a34e42170. The exact released processor and module
bodies execute unchanged. Processor BF16 patches are promoted to FP32 at the
declared qualification boundary. The native graph uses fused QKV and SwiGLU
projections, 2D RoPE, spatial pad/unfold, GPU backward and CPU-policy AdamW on
Vulkan. No native output supplies expected CPU values.

This is a component receipt. It does not qualify the full V4.1 multimodal
language pipeline or promote the 44th profile. Image-span merge, delimiter
training, image-specific MoE routing, joint LoRA updates, and export/reload
remain required for that promotion. Language base parameters remain frozen.

See qualification.json, provenance.json and component-tests.log for counts,
binary/source hashes and all measured rows. Historical failed attempts remain
under target/multimodal-gen9/v41-native-vision-attempt1 through attempt5.
Physical RDNA3 replay is pending.

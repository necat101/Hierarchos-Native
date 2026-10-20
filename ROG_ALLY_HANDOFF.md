# Intel validation and ROG Ally qualification

This candidate contains the Intel Gen9 producer routes, portable recurrent/TBPTT
optimizations, and the upstream two-rank distributed trainer with a Windows TCP
transport. AMD routes must be measured on the physical Ally. Intel measurements
are keyed by adapter/shape/storage policy and are not AMD tuning results.
The four RDNA3 vendor matmul modules (eight GLSL/SPIR-V files) are additions
relative to `b519361`. Their physical execution is pending the Ally run; the
module inventory in the Intel results records their hashes. The AMD gate must
exercise those modules and compare the candidate against stable on that device.

## Clone the published handoff

Clone the verified candidate and create its matching clean reference checkout:

```powershell
git clone --branch codex/verified-intel-ally-handoff https://github.com/necat101/Hierarchos-Native.git
Set-Location Hierarchos-Native
git worktree add --detach ..\Hierarchos-windows-oracle origin/codex/stable-b519361-windows-oracle
```

The oracle branch is the exact Windows build-only reference used for Intel
qualification. After provisioning the validation environment described below:

```powershell
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan\validation\qualify_handoff.py `
  --stable ..\Hierarchos-windows-oracle --expected-vendor amd `
  --output .\target\ally-results --tune
```

The archived Intel evidence and exact dependency versions are in
`hierarchos-vulkan/validation/intel-handoff/`. Build on the Ally and keep its
fresh receipts separate from the Intel evidence.

## Local qualification result and one existing numerical exception

All 13 Intel qualification stages passed. The 75 registered Transformer cases
compare 36,096 FP32 logits with zero differing bits; the recurrent comparison
checks 8,067 exact tensor digests under both storage policies, including 162
checkpoint-reloaded logit vectors over 27 optimizer updates. Debug and release
training ladders each pass all 12 checks and 32,162 exact trajectory digests.
Both two-process TCP trainer modes produce identical model/optimizer bytes for
all 290 exported tensors. Original tolerances and numerical goldens are unchanged.

**This is not unrestricted parity for every architecture.** The pre-existing
`transformer_qwen_gdn_params.spv` repair applies OLMo Hybrid's requested beta
scale 2; the old stable artifact ignores it. This is documented in
`hierarchos-vulkan/validation/oracle_divergences.json` and matches the clean
Transformers source's `beta = beta * 2.0` contract. A separate two-layer fixture
has 282/282 differing logits, maximum absolute difference `0.84882572`, with the
feature enabled. Disabling it makes all 282 logits bit-identical. The 75-case
registry does not contain OLMo Hybrid. The user explicitly approved retaining
this correctness repair as a documented exception to old-stable parity. The
diagnostic package and results are included.

## Stable reference

The live stable checkout is `b519361`. Its unconditional Linux libibverbs build
fails on Windows (`infiniband/verbs.h` is missing). Keep that checkout unchanged.
For numerical comparison, make a separate reference:

```powershell
git -C C:\path\to\stable worktree add --detach C:\path\to\windows-oracle b519361
git -C C:\path\to\windows-oracle apply C:\path\to\candidate\hierarchos-vulkan\validation\windows_oracle_adapter.patch
git -C C:\path\to\windows-oracle add hierarchos-vulkan/build.rs hierarchos-vulkan/src/lib.rs hierarchos-vulkan/src/distributed_windows_unavailable.rs
git -C C:\path\to\windows-oracle commit -m "Windows oracle build adapter: disable unavailable remote RDMA only"
```

The patch disables remote RDMA on the **reference only**. It changes no shader,
optimizer math, inference implementation, training graph, or initialization.
The candidate implements functional TCP instead. Receipts record the actual
reference commit, its source fingerprint, and the candidate's dirty source
fingerprint. A comparison against the parent `58632d3` is additional evidence,
not a substitute for this live-reference comparison.

## Run on the Ally

Use the same validation Python environment and adjacent Transformers source
checkout as the Intel host (including its local K3 oracle dependencies), Rust
MSVC toolchain, Git Bash, and Vulkan driver. Set
`HIERARCHOS_TRANSFORMERS_CHECKOUT` if Transformers is not adjacent. The training
ladder resolves/installs its pinned shader toolchain under `.tools/`.
Build locally on the Ally; do not transfer binaries compiled with
`-C target-cpu=native` from another CPU.

From the candidate repository root, on a quiet adapter:

```powershell
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan\validation\qualify_handoff.py `
  --stable C:\path\to\windows-oracle --expected-vendor amd `
  --output C:\path\to\ally-results --tune
```

The runner first checks Python contracts, all binaries, library/trainer/frontends,
two-rank TCP loopback, both recurrent storage policies against stable, all registry
logits against stable and Transformers, actual RDNA3 vendor-module dispatches,
the independent CPU RWKV bridge, and debug/release training ladders. Only after
every gate passes does `--tune` measure and merge this adapter's producer/chain
routes, rebuild, and repeat numerical qualification. Route recording and receipt
recording are separate passes. Intel rows are retained.
If any tuning measurement or subsequent parity gate fails, the runner restores
the pre-tuning route tables and rebuilds their binaries. Failed measurements
remain in the output logs for review.

`qualification.json` is green only if every command exits zero. Every step has
its own log. Unrecorded device goldens fail explicitly; never re-record a
candidate result merely to turn that failure green. Compare that device against
stable first and establish reference-derived goldens separately. No gate widens
a numerical tolerance or re-records a numerical golden.

The direct stable logit gate compares all FP32 output bits, including masked
positions and signed zeros. The independent Transformers gate retains its
absolute `2e-7` ceiling. Recurrent comparison covers parameters, moments/clocks,
losses, states/adjoints, context carriers, and six inference-logit vectors after
every exported optimizer update, under FP32 and FP16-storage/FP32-compute.
The wide Transformer fixtures use twelve rows in both cases, clearing RDNA3's
eight-row vendor-kernel threshold as well as Intel's gate. Dispatch counters
must confirm that the planned tuned modules actually ran.

## Windows distributed training

Local multi-GPU training still uses `--device-indices`. Cross-machine training
currently supports exactly two ranks, one device per rank. Run the same dataset,
model/configuration, seed, precision and optimizer settings on both machines;
use accumulation windows with at least two microbatches, including the last
window. On a trusted LAN, allow the trainer's two TCP ports (18515/18516; verify
the constants in `train.rs`) through the Windows firewall.

```powershell
# Primary: replace PEER with the replica's address and MODEL/DATA/OUTPUT paths.
cargo run --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train -- `
  --model MODEL --dataset DATA --output OUTPUT --batch-size 1 `
  --gradient-accumulation-steps 2 --world-size 2 --rank 0 --peer-ip PEER `
  --network-transport tcp --distributed-mode param-server
# Replica: same command with --rank 1, peer address pointing to the primary,
# and a separate output directory. allreduce is also supported.
```

The primary broadcasts initial parameters, moments/clocks and controller state
in both modes. Param-server mode reduces gradients on the primary and broadcasts
the updated state. Allreduce exchanges raw gradients, applies the optimizer on
each rank and fails on a weight-checksum mismatch. Logs identify `network-tcp`.
`HIERARCHOS_NETWORK_TIMEOUT_SECONDS` controls socket timeouts (default 300).
Tensor/control sockets are separate, frames are bounded, short tensor tails are
supported, and simultaneous large sends receive concurrently to avoid deadlock.

TCP preserves the training protocol, but does not provide RDMA performance or
GPU-direct networking. Linux RDMA remains an opt-in `--features rdma` build with
libibverbs headers/libraries and a real RDMA NIC. Windows native RDMA uses a
different [NetworkDirect interface](https://github.com/microsoft/NetworkDirect);
it is not implemented or certified by this TCP port.

The port also repairs upstream distributed-state defects: the decoder now reads
all first Adam moments before all second moments, matching the wire encoder;
unaligned payloads are decoded without pointer casts; metadata length/overflow
checks fail before indexing; both modes synchronize initial optimizer state;
and an allreduce weight mismatch fails the run. The transport regression test
checks the moment round-trip with nonzero optimizer state.

The GPU transport test compares the existing local streamed reduction with TCP,
including model/optimizer bytes and checkpoint-reloaded logits. Two-process
loopback tests exercise both trainer modes and compare all exported model/optimizer
tensor bytes between param-server and allreduce. Distributed per-rank gradient
sums group additions differently from monolithic per-use accumulation; FP32
non-associativity means these are separate numerical contracts. No tolerance is
introduced to hide that distinction. The tests do not certify two
physical GPUs, a real network, native RDMA hardware, or AMD performance. Those
hardware checks remain part of the Ally/second-machine handoff.

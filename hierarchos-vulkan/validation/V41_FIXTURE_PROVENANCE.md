# V4.1 fixture provenance on Intel Gen9

The i5-6200U runs torch **2.14.1+cpu**, numpy **2.5.3**, CPU capability
**AVX2**. These are measured interpreter properties, not inferred from the GPU.
`fixture_toolchain.py` records all three plus the SHA-256 of the exact fixture
bytes in `fixtures/deepseek_v41_toolchain.json`.

A pin is created only when a generator produces a new fixture or reproduces
the complete payload of an existing one. Existing bytes are preserved even
when JSON whitespace differs. A changed payload is refused. A stale sidecar
hash, conflicting inline pin, or a different interpreter/CPU fails explicitly.
`--allow-toolchain-mismatch` permits a diagnostic comparison with a warning;
it does not replace frozen values or silently reassign an existing pin.

The V4.1 numerical generators and the production attention trace
use this helper. Fixed fixture paths check their pins before generation, and
every frozen fixture write checks again. Observation files and generated model
packages are separate from frozen numerical fixtures.

## Verified reproductions

The seven historical fixtures reproduced here without changing any of their bytes:

- rotary;
- quantization;
- BF16 FP4 midpoint quantization;
- index keys;
- index selection;
- candidates;
- adapter training, including all three optimizer steps.

The new `deepseek_v41_mixed_block128.json` has two packed linear cases (642
outputs) and one nine-row MoE case (1,152 outputs). Its model/kernel sources are
the existing SHA-256-verified release at `1bae0e1088993b9b82de08f59de5be5a34e42170`.
`deepseek_v41_terminal_grad.avx2.json` is a separately generated CPU-autograd
oracle. The terminal test selects it for the host's eight-lane ATen policy;
the original terminal fixture remains unchanged for the sixteen-lane policy.

There are now **17 verified pins**. They include nine AVX2 siblings and the new
block128 fixture. The **54 other historical
fixtures have no verified toolchain pin**. Their exact hashes remain in
`deepseek_v41_toolchain_legacy.json`; that list is a compatibility baseline,
not proof of reproduction on AVX-512 or any other machine. The CPU difference
can affect reduction rounding, but it does not establish which toolchain
originally generated a particular file.

## Checks and regeneration

CI audits the canonical library pair without importing torch and rejects new
unpinned files, changed legacy files, stale hashes, mismatched pins and baseline
entries that should have been retired:

Legacy entries must also match the original `HANDOFF-MANIFEST.json`, so adding
a new fixture to the legacy list cannot bypass the pin requirement.

```powershell
python hierarchos-vulkan/validation/fixture_toolchain.py check-all --expect-from hierarchos-vulkan/validation/fixtures/deepseek_v41_mixed_block128.json
python -m unittest discover -s hierarchos-vulkan/validation -p test_fixture_toolchain.py
```

Add `--strict` to require every historical fixture to have a verified pin. It
currently fails on the 54 legacy entries. Reproducing an existing fixture with
its generator's write mode pins it and removes only its own baseline entry.
Never retire an entry by guessing the original host's versions.

For a justified CPU-specific output, set `HIERARCHOS_FIXTURE_VARIANT=avx2`
while invoking that generator. It writes a sibling, leaving the original
fixture untouched. This does not automatically make a kernel compatible with
that oracle: the numerical test must select the corresponding host schedule
and still meet its original tolerance. The Rust gates select the AVX2 terminal,
loss-reduction and full BF16-training siblings for the eight-lane policy.
The other complete-trajectory siblings remain diagnostics: their different
arithmetic order is not interchangeable with the frozen forward contract.

`deepseek_v41_training_loss.avx2.json` separately binds 31 CPU loss scalars to
every original logit and target word with SHA-256. This lets the host's
eight-lane CE kernel use a genuine CPU reference while keeping the original
projection/model logits, gradients, factors and moments unchanged. The GPU
probe reproduces every checked row loss and final scalar bit for bit.

The inference and mixed-MoE gates keep their original fixtures. In particular,
the compressor retains sixteen-channel vector bodies and scalar tails, and
short, wide projection adjoints retain the frozen four-product groups. These
operator schedules do not change when the serving CPU reports AVX2.

The five retained structure/wide packages can be restored without executing
their CPU forward trajectories. The reference directory must also contain the
SHA-256-checked released `engram.py` used by the existing model generator:

```powershell
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/restore_deepseek_v41_checkpoints.py --reference-path .tools/v41-reference/model.py --kernel-reference-path .tools/v41-reference/kernel.py
```

Every generated file must match the original fixture's checkpoint hash before
it is written. All 18 file hashes match; no numerical fixture is regenerated
or assigned a toolchain pin by this restoration command.

Example, after providing the two pinned reference sources:

```powershell
.\.venv-vulkan\Scripts\python.exe hierarchos-vulkan/validation/validate_deepseek_v41_mixed.py --reference-path .tools/v41-reference/model.py --kernel-reference-path .tools/v41-reference/kernel.py --block128
```

Repeating that command compares the full payload. `--write-fixture` creates a
new target or verifies/pins an unchanged one; it refuses numerical replacement.

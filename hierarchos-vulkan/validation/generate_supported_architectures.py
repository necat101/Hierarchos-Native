#!/usr/bin/env python3
"""Generate the native Vulkan architecture README from the Rust registry."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _ordered_registry(transformer_rs: str) -> tuple[list[str], list[str]]:
    model_type_match = re.search(
        r"pub fn model_type\(self\).*?match self \{(.*?)\n\s*\}\n\s*\}",
        transformer_rs,
        re.DOTALL,
    )
    if model_type_match is None:
        raise RuntimeError("could not locate VulkanTransformerArchitecture::model_type")
    canonical = re.findall(r'=>\s*"([^"]+)"', model_type_match.group(1))

    alias_match = re.search(
        r"HF_MODEL_TYPE_ALIASES.*?=\s*&\[(.*?)\];",
        transformer_rs,
        re.DOTALL,
    )
    if alias_match is None:
        raise RuntimeError("could not locate HF_MODEL_TYPE_ALIASES")
    aliases = re.findall(r'"([^"]+)"', alias_match.group(1))

    if len(canonical) != len(set(canonical)):
        raise RuntimeError("canonical model_type registry contains duplicates")
    if len(aliases) != len(set(aliases)):
        raise RuntimeError("alias model_type registry contains duplicates")
    overlap = set(canonical) & set(aliases)
    if overlap:
        raise RuntimeError(f"canonical/alias registry overlap: {sorted(overlap)}")
    return canonical, aliases


def _table(values: list[str], columns: int = 4) -> str:
    values = sorted(values)
    rows = (len(values) + columns - 1) // columns
    padded = values + [""] * (rows * columns - len(values))
    headers = [f"model_type {index + 1}" for index in range(columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * columns) + " |",
    ]
    for row in range(rows):
        cells = []
        for column in range(columns):
            value = padded[row + column * rows]
            cells.append(f"`{value}`" if value else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(canonical: list[str], aliases: list[str]) -> str:
    advertised = len(canonical) + len(aliases)
    return f"""# Hierarchos Vulkan supported Transformer architectures

This file is generated from `src/transformer.rs`. Do not hand-maintain the
architecture counts or lists; regenerate it with:

```powershell
python hierarchos-vulkan/validation/generate_supported_architectures.py
```

The current native registry contains **{len(canonical)} canonical Transformer
architectures** plus **{len(aliases)} Hugging Face package/config aliases**
({advertised} advertised `model_type` spellings total).

## What "end-to-end" means here

The canonical entries below are the native Vulkan Transformer's declared text
graph contracts. They are routed through architecture-specific Hugging Face
configuration parsing, SafeTensors loading, Vulkan forward/backward/training,
generation where the architecture is generative, and native package export.
Unsupported `model_type` values fail closed; the backend does not silently map
an unknown architecture onto a merely similar graph.

This is deliberately narrower than "all Transformers tasks." A text graph being
supported does not imply that every upstream task head, vision tower, audio
tower, processor, custom remote-code model, or multimodal projector is native.
The aliases section is especially important: those names resolve a supported
native **text backbone** from a composite Hugging Face package. They are not a
claim that the package's non-text modalities execute end to end in Vulkan.

For registry coverage against the checked-out Python `transformers` source, run
`python hierarchos-vulkan/validation/audit_transformers_coverage.py`. That audit
is registry overlap, not a numerical-parity certificate. Parity and training
validation live under `hierarchos-vulkan/validation/`.

## User-facing native routes

| Workflow | CLI | Desktop GUI |
| --- | --- | --- |
| Transformer full training | `transformer-train` | Transformer full training |
| Transformer LoRA fine-tuning | `transformer-finetune` | Transformer LoRA fine-tuning |
| Transformer inference/generation | `infer` / `generate` / `transformer-generate` | Transformer inference / generation |
| Hierarchos training | `train` / `finetune` | Hierarchos training / fine-tuning |
| Hierarchos inference | `chat` | Hierarchos inference / chat |

The GUI is a thin native launcher over the same CLI contracts; it does not carry
a second model implementation. Transformer training and generation execute in
the Vulkan backend. Hierarchos training executes in Vulkan and Hierarchos
inference uses the companion pure-Rust inference engine.

## Canonical native architectures ({len(canonical)})

{_table(canonical)}

## Hugging Face package/config aliases ({len(aliases)})

These aliases resolve to a canonical native text graph while preserving
unconsumed package tensors on export where supported.

{_table(aliases)}

## Shipping checks

Before publishing a standalone bundle, run at least:

```powershell
cargo check --manifest-path hierarchos-vulkan/Cargo.toml --all-targets
cargo test --manifest-path hierarchos-vulkan/Cargo.toml --lib
cargo test --manifest-path hierarchos-native-cli/Cargo.toml
cargo test --manifest-path hierarchos-gui/Cargo.toml
python -m unittest discover -s hierarchos-vulkan/validation -p "test_*.py"
python hierarchos-vulkan/validation/generate_supported_architectures.py --check
hierarchos-native-cli architectures --json
hierarchos-native-cli doctor
```

`doctor` verifies Vulkan device visibility and reports whether the companion
Hierarchos trainer/device-probe binaries can be found. Transformer training and
generation are implemented directly in `hierarchos-native-cli`; coherent-v9
Hierarchos training uses the companion `hierarchos-vulkan-train` binary.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if README_ARCHITECTURES.md is not current instead of rewriting it",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    transformer_rs = repo_root / "hierarchos-vulkan/src/transformer.rs"
    output = repo_root / "hierarchos-vulkan/README_ARCHITECTURES.md"
    canonical, aliases = _ordered_registry(transformer_rs.read_text(encoding="utf-8"))
    rendered = render(canonical, aliases).replace("\n", "\r\n")

    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered.replace("\r\n", "\n"):
            print(f"stale generated architecture README: {output}", file=sys.stderr)
            return 1
        print(f"architecture README is current: {len(canonical)} canonical + {len(aliases)} aliases")
        return 0

    output.write_text(rendered, encoding="utf-8", newline="")
    print(f"wrote {output}: {len(canonical)} canonical + {len(aliases)} aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

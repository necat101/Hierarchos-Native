"""Restore the small frozen structure/wide packages without a CPU forward run.

Only model construction and the original weight transforms supply package
bytes. Every generated file must match the original fixture's SHA-256 before
any file is written. Logits, routing expectations and toolchain pins are never
modified by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path

import validate_deepseek_v41_structure_model as structure
import validate_deepseek_v41_wide_model as wide


def restore(fixture, files, root):
    expected = fixture["checkpoint"]["files"]
    actual = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if actual != expected:
        raise AssertionError(f"seed {fixture['seed']}: checkpoint hashes differ")
    destination = root / fixture["checkpoint"]["directory"]
    for name, data in files.items():
        if Path(name).name != name:
            raise ValueError(f"unexpected checkpoint path: {name}")
        path = destination / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"existing checkpoint file differs: {path}")
    destination.mkdir(exist_ok=True)
    for name, data in files.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(data)
    print(f"PASS seed {fixture['seed']}: {len(files)} frozen file hashes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "fixtures"
    # Empty steps affect only execution. The package hash check independently
    # proves that construction and every weight transform remain unchanged.
    original_inputs = wide.inputs
    wide.inputs = lambda vocab, schedule=None: ([], original_inputs(vocab, schedule)[1])
    try:
        for module, name in [(structure, "structure"), (wide, "wide")]:
            frozen = json.loads((root / f"deepseek_v41_{name}_model.json").read_text())
            for spec, fixture in zip(module.SPECS, frozen["cases"]):
                generated = module.make_case(args.reference_path, args.kernel_reference_path,
                                             spec, None)
                restore(fixture, wide.checkpoint_bytes(generated, spec.get("sharded", True)), root)
    finally:
        wide.inputs = original_inputs


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare every registry logit against a clean stable checkout, bit for bit.

Both binaries load the same checkpoint and input JSON. The candidate also has
to pass the existing independent Transformers oracle at its unchanged 2e-7 gate.
The stable comparison includes ALL output positions, including masked positions,
and compares finite FP32 bits (so signed zero counts). No tolerance is available.
Run on each physical adapter; this is not a cross-driver bit-identity claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
BINARY = "hierarchos-vulkan-transformer-logits"


def identity(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    digest = hashlib.sha256()
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z",
                                     "--cached", "--others", "--exclude-standard"])
    for name in sorted(set(paths.decode().split("\0")) - {""}):
        path = root / name
        if path.suffix in {".rs", ".spv", ".comp", ".toml", ".json", ".py", ".lock"}:
            digest.update(name.encode() + b"\0")
            digest.update(path.read_bytes() if path.exists() else b"MISSING")
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain"), "source_sha256": digest.hexdigest()}


def build(crate: Path, profile: str, binary: str = BINARY,
          target_directory: Path | None = None) -> Path:
    command = ["cargo", "build", "--manifest-path", str(crate / "Cargo.toml"),
               "--bin", binary, "--message-format=json-render-diagnostics"]
    if target_directory is not None:
        command += ["--target-dir", str(target_directory), "-j", "1"]
    if profile == "release":
        command.append("--release")
    completed = subprocess.run(command, cwd=crate, capture_output=True, text=True, check=True)
    for line in completed.stdout.splitlines():
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact" and message.get("executable")
                and message["target"]["name"] == binary):
            return Path(message["executable"])
    raise RuntimeError(f"Cargo reported no {binary} in {crate}")


def compare(reference: dict, candidate: dict) -> dict:
    for key in ("device", "architecture", "batch_size", "seq_len", "vocab_size"):
        if key not in reference or reference[key] != candidate.get(key):
            raise ValueError(f"logit report {key} differs or is missing")
    left, right = reference.get("logits"), candidate.get("logits")
    if not isinstance(left, list) or not left or not isinstance(right, list) or len(left) != len(right):
        raise ValueError("empty, missing, or different logit counts")
    expected = reference["batch_size"] * reference["seq_len"] * reference["vocab_size"]
    if len(left) != expected:
        raise ValueError(f"logit count {len(left)} != shape product {expected}")
    differing = []
    digest = hashlib.sha256()
    max_abs = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"non-finite logit at {index}")
        a_bits, b_bits = struct.pack("<f", a), struct.pack("<f", b)
        max_abs = max(max_abs, abs(struct.unpack("<f", a_bits)[0] - struct.unpack("<f", b_bits)[0]))
        digest.update(a_bits)
        if a_bits != b_bits:
            differing.append(index)
    return {"values": len(left), "different_bits": len(differing),
            "first_differences": differing[:16],
            "max_abs": max_abs,
            "reference_fp32_sha256": digest.hexdigest(), "pass": not differing}


def comparisons_pass(rows: list[dict]) -> bool:
    """Neither stable bit identity nor the independent HF gate can waive the other."""
    return bool(rows) and all(row["pass"] and row["hf_failing_values"] == 0 for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True, type=Path)
    parser.add_argument("--profile", choices=("debug", "release"), default="release")
    parser.add_argument("--output", type=Path, default=CRATE / "target" / "stable-logits")
    args = parser.parse_args()
    stable = args.stable.resolve()
    if stable == CRATE.parent or not (stable / "hierarchos-vulkan" / "Cargo.toml").is_file():
        parser.error("--stable must name a separate Hierarchos repository")
    baseline_identity = identity(stable)
    if baseline_identity["status"]:
        parser.error("the stable oracle must be clean; its sources will never be edited")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Imported here so negative controls have no torch/Transformers dependency.
    import verify_hf_logits as registry
    transformers_source = Path(registry.transformers.__file__).resolve()
    if not transformers_source.is_relative_to(registry.TRANSFORMERS_SOURCE):
        raise ValueError(f"unexpected Transformers oracle source: {transformers_source}")

    candidate_identity = identity(CRATE.parent)
    current_binary = build(CRATE, args.profile)
    stable_binary = build(stable / "hierarchos-vulkan", args.profile)
    # Preserve the legacy comparison separately. Only GPT-2 may use this
    # minimal, independently rebuilt repair reference; all other families keep
    # the clean stable executable as their exact-bit gate.
    import numerical_repair_oracle as repair
    repaired_crate, repair_metadata = repair.prepare(stable, CRATE,
        CRATE / "target" / "numerical-repair-oracle" / "source")
    repaired_binary = build(repaired_crate, args.profile, repair.BINARY,
        stable / "hierarchos-vulkan" / "target")
    registry.torch.manual_seed(0x48494552)
    models = registry.tiny_models() + registry.production_width_models()
    names = [name for name, _, _ in models]
    if len(names) != len(set(names)) or set(registry.STRICT_LOGIT_FAMILIES) - set(names):
        raise ValueError("registry fixtures duplicate a family or omit a strict-logit family")
    expected_cases = {(name, case) for name in names
                      for case in (("unmasked",) if name in registry.KIMI_K3_ORACLE_FAMILIES
                                   else ("unmasked", "padded"))}
    if len(expected_cases) < 75:
        raise ValueError("registry coverage shrank below the 75 qualified comparisons")
    rows = []
    for name, model, token_types in models:
        padded_cases = (False,) if name in registry.KIMI_K3_ORACLE_FAMILIES else (False, True)
        for padded in padded_cases:
            case = "padded" if padded else "unmasked"
            print(f"Comparing {name}/{case}", flush=True)
            hf = registry.compare_family(current_binary, args.output / "fixtures", name,
                                         model, token_types, None, 2e-7, 0.0, padded,
                                         raise_on_mismatch=False)
            model_dir = args.output / "fixtures" / name / case
            current_report = json.loads((model_dir / "vulkan_logits_report.json").read_text())
            stable_report_path = model_dir / "stable_logits_report.json"
            subprocess.run([str(stable_binary), "--model", str(model_dir), "--fixture",
                            str(model_dir / "vulkan_logits_fixture.json"), "--output",
                            str(stable_report_path)], cwd=stable, check=True,
                           capture_output=True, text=True)
            legacy = compare(json.loads(stable_report_path.read_text()), current_report)
            exact = legacy
            if name in repair.REPAIRED_FAMILIES:
                corrected_path = model_dir / "corrected_stable_logits_report.json"
                subprocess.run([str(repaired_binary), "--model", str(model_dir), "--fixture",
                    str(model_dir / "vulkan_logits_fixture.json"), "--output", str(corrected_path)],
                    cwd=repaired_crate, check=True, capture_output=True, text=True)
                exact = compare(json.loads(corrected_path.read_text()), current_report)
            row = {"family": name, "case": case, "device": current_report["device"],
                   "hf_max_abs": hf["max_abs"], "variant_dispatches": hf["variant_dispatches"],
                   "hf_failing_values": hf["failing_values"],
                   "reference": "stable-plus-numerical-repair" if name in repair.REPAIRED_FAMILIES else "clean-stable",
                   "legacy_stable": legacy, **exact}
            rows.append(row)
            print(f"  {row['values']} logits, {row['different_bits']} differing bits", flush=True)
    if {(row["family"], row["case"]) for row in rows} != expected_cases:
        raise ValueError("registry comparison coverage is incomplete")
    if identity(stable) != baseline_identity or identity(CRATE.parent) != candidate_identity:
        raise ValueError("source tree changed during validation; rerun on a quiet checkout")
    repair.verify_prepared(repaired_crate, repair_metadata)
    receipt = {"result": "pass" if comparisons_pass(rows) else "fail",
               "stable": baseline_identity, "candidate": candidate_identity,
               "profile": args.profile, "comparisons": len(rows),
               "transformers_source": str(transformers_source),
               "transformers_version": registry.transformers.__version__,
               "values": sum(row["values"] for row in rows),
               "different_bits": sum(row["different_bits"] for row in rows), "rows": rows,
               "legacy_stable_different_bits": sum(row["legacy_stable"]["different_bits"] for row in rows),
               "numerical_repair": repair_metadata,
               "binaries": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                            for label, path in (("stable", stable_binary), ("candidate", current_binary),
                                                ("stable_plus_repair", repaired_binary))}}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{receipt['result'].upper()}: {receipt['comparisons']} comparisons, "
          f"{receipt['values']} logits, {receipt['different_bits']} differing bits")
    return int(receipt["result"] != "pass")


if __name__ == "__main__":
    sys.exit(main())

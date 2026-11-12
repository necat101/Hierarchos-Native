#!/usr/bin/env python3
"""Freeze fresh two-pass attention-backward evidence on the current source.

The full Phase XVI receipt covers 16 GPT-2 shapes and 74 retained cases. Its
CPU oracle inputs are not present in this checkout, so it remains a historical
receipt. This producer instead runs the current-source two-pass backward test
and dispatch harness, binds those logs to the current trainer and shader bytes,
and records the Phase XVI shape/retained coverage as historical metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from emit_two_pass_evidence import ENVIRONMENT, parse

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19x" / "attention"
)
OLD = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261110-phase19s-current-source" / "attention" / "current-qualification.json"
)
WORK = CRATE / "target/phase19x-attention/two-pass"
GATE = CRATE / "target/phase19s/run/phase16-spirv.json"
SHADERS = CRATE / "shaders"
SOURCE_FILES = (
    "src/transformer.rs",
    "src/lora.rs",
    "src/transformer/peft_state.rs",
    "src/bin/transformer_train.rs",
    "validation/two_pass_backward_equivalence.sh",
    "validation/qualify_attention_phase19x.py",
    "validation/check_attention_phase19x.py",
    "validation/test_attention_phase19x.py",
)


def require(condition, message):
    if not condition:
        raise SystemExit(f"ATTENTION-XIXX PRODUCER: FAIL: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty or args.allow_dirty, "tracked worktree is dirty:\n" + dirty)
    revision = git("rev-parse", "HEAD")
    require(len(revision) == 40, "cannot resolve revision")

    require(WORK.is_dir(), f"missing {WORK}")
    gradient = parse(WORK / "test.log", WORK / "harness.log")
    require(gradient["pass"], "two-pass evidence failed: " + "; ".join(gradient["failures"]))
    require(gradient["cases"] and len({row["case"] for row in gradient["cases"]}) == 4,
            "two-pass evidence does not cover four cases")
    require(gradient["comparisons"] == 12, "two-pass evidence does not cover twelve tensors")

    old = load(OLD)
    require(old.get("pass") is True, "historical Phase XVI receipt is not a pass")
    gate = load(GATE)
    require(gate.get("pass") is True and gate.get("counts", {}).get("total") == 45,
            "45-artifact shader gate changed")
    require((gate.get("status_counts") or {}).get("identical") == 45,
            "shader gate is not 45 identical artifacts")
    sealed = {row["artifact"]: row["sha256"] for row in gate["artifacts"]
              if row.get("origin") == "sealed"}
    rebuilt = {row["artifact"]: row["sha256"] for row in gate["artifacts"]
               if row.get("origin") == "phase16"}
    require(len(sealed) == 43 and len(rebuilt) == 2,
            "shader gate does not contain 43 historical and 2 Phase XVI artifacts")
    for name, expected in (*sealed.items(), *rebuilt.items()):
        require((SHADERS / name).is_file(), "missing shader " + name)
        require(digest(SHADERS / name) == expected, "shader changed: " + name)

    source_files = {}
    for name in SOURCE_FILES:
        path = CRATE / name
        require(path.is_file(), "missing source " + name)
        source_files[name] = digest(path)

    trainer = CRATE / "target/release/hierarchos-vulkan-transformer-train.exe"
    require(trainer.is_file(), "missing current trainer")

    evidence_dir = WORK / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)
    evidence = {}
    for name, source in (("two-pass/test.log", WORK / "test.log"),
                         ("two-pass/harness.log", WORK / "harness.log"),
                         ("phase16-spirv.json", GATE)):
        destination = evidence_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        evidence[name] = digest(destination)

    backward = {
        "enabled_by": ENVIRONMENT,
        "cases": len({row["case"] for row in gradient["cases"]}),
        "comparisons": gradient["comparisons"],
        "elements": gradient["elements"],
        "bitwise_equal": gradient["bitwise_equal"],
        "max_abs": gradient["max_abs"],
        "max_ulp": gradient["max_ulp"],
        "unwritten": gradient["unwritten"],
        "adapters_identical": gradient["adapters"]["identical"],
        "adapter_files": [f"{path} {sha}" for path, sha in gradient["adapters"]["files"]],
        "dispatches": gradient["dispatches"],
        "diagnostic": gradient["diagnostic"],
        "negative_controls": gradient["negative_controls"],
    }
    backward["test_log_sha256"] = gradient["evidence"]["test_log"]
    backward["harness_log_sha256"] = gradient["evidence"]["harness_log"]

    retained = old.get("retained") or {}
    document = {
        "schema": 1,
        "kind": "phase19x-two-pass-attention-current-source",
        "status": "PASS",
        "pass": True,
        "target": "rdna3",
        "scope": "phase19x_two_pass_attention_backward_current_source",
        "source_revision": revision,
        "source_clean": True,
        "source_files": source_files,
        "binaries": {
            "trainer": digest(trainer),
        },
        "attention_backward": backward,
        "shaders": {
            "inventory": 45,
            "status_counts": {"identical": 45},
            "historical_artifacts": 43,
            "phase16_artifacts": 2,
            "sealed_gate_sha256": digest(GATE),
        },
        "historical_phase16": {
            "receipt_sha256": digest(OLD),
            "source_head": old.get("source_head"),
            "shapes": old.get("shapes", {}).get("cases"),
            "retained": {name: row.get("cases") for name, row in retained.items()},
            "note": "Full Phase XVI shape and retained-family evidence remains historical; it was not re-executed because the AVX-512 CPU oracle inputs are absent.",
        },
        "evidence": evidence,
        "limitations": [
            "Only the four-case, twelve-tensor two-pass backward is current-source evidence.",
            "The 16-shape and 74-retained-case Phase XVI evidence is historical and is not claimed as current-source evidence.",
            "Raw GPU captures and adapter files remain under target/ and are bound by hash only.",
        ],
    }
    if args.install:
        if RECEIPT.exists():
            raise SystemExit(f"refusing to overwrite existing receipt {RECEIPT}")
        RECEIPT.mkdir(parents=True)
        (RECEIPT / "current-qualification.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8")
        for name in evidence:
            destination = RECEIPT / "evidence" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(evidence_dir / name, destination)
    else:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "receipt.json").write_text(json.dumps(document, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"ATTENTION-XIXX: PASS ({revision[:12]}, {backward['comparisons']} tensors, "
          f"{backward['elements']} elements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

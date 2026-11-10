#!/usr/bin/env python3
"""Verify the Phase XVI current-source qualification for the two-pass backward.

This is the current-source gate. It accepts only a receipt whose bound source
bytes still match this checkout, whose evidence files are present and hash to
what the receipt records, and whose graded fields re-derive from the raw logs
it ships. Phase XV's sealed receipt is history: it is read for the 43-artifact
identities and never accepted as current evidence.

A missing receipt is an unqualified source (exit 3), not a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# Source bindings are repository-relative; shaders, receipts and the sealed gate
# live beside this crate's validation directory.
ROOT = Path(__file__).resolve().parents[2]
CRATE = Path(__file__).resolve().parents[1]
VALIDATION = Path(__file__).resolve().parent
RECEIPT_DIR = (
    VALIDATION / "release_parity" / "multimodal_training"
    / "20261109-rdna3-phase16-attention-backward-qualification"
)
CURRENT_RECEIPT = RECEIPT_DIR / "current-qualification.json"
SEALED_GATE = (
    VALIDATION / "release_parity" / "multimodal_training"
    / "20261108-rdna3-phase15-qualification" / "gates" / "spirv.json"
)
SHADERS = CRATE / "shaders"

SCOPE = "phase16_two_pass_attention_backward_and_retained_parity"
ENVIRONMENT = "HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS"
LIMIT = 2e-7
COUNTS = {"shapes": 16, "shared": 74, "kimi": 3, "seq2seq": 6, "gemma3": 4, "gemma4": 2, "llava": 4}
GATES = ("shared", "kimi", "seq2seq", "gemma3", "gemma4", "llava")
PHASE16_ARTIFACTS = (
    "transformer_attention_backward_query_rdna3.spv",
    "transformer_attention_backward_key_rdna3.spv",
)
# The two-pass work touched these files; a receipt that does not bind them
# cannot describe the implementation it claims to qualify.
EXTRA_SOURCES = (
    "hierarchos-vulkan/shaders/transformer_attention_backward.comp",
    "hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.comp",
    "hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.spv",
    "hierarchos-vulkan/shaders/transformer_attention_backward_key_rdna3.comp",
    "hierarchos-vulkan/shaders/transformer_attention_backward_key_rdna3.spv",
    "hierarchos-vulkan/validation/check_phase16_receipts.py",
    "hierarchos-vulkan/validation/emit_phase16_spirv_gate.py",
    "hierarchos-vulkan/validation/emit_two_pass_evidence.py",
    "hierarchos-vulkan/validation/qualify_phase16_attention_backward.py",
    "hierarchos-vulkan/validation/test_phase16_receipts.py",
    "hierarchos-vulkan/validation/two_pass_backward_equivalence.sh",
    "hierarchos-vulkan/validation/training_shader_manifest.json",
    "hierarchos-vulkan/src/transformer.rs",
    "hierarchos-vulkan/src/vulkan.rs",
)
REQUIRED_KEYS = (
    "schema", "pass", "target", "scope", "source_head", "worktree", "absolute_limit",
    "rtol", "comparison", "adapter", "binaries", "producer_sha256",
    "cpu_fixtures_sha256", "shapes", "retained", "primitives", "v41", "shaders",
    "attention_backward", "isolation", "sources", "evidence", "reused", "limits",
)
REQUIRED_EVIDENCE = (
    "shapes/qualification.json", "shared/qualification.json", "kimi/qualification.json",
    "seq2seq/qualification.json", "gemma3/qualification.json",
    "gemma4-legacy/qualification.json", "gemma4-rgb/qualification.json",
    "llava/qualification.json", "primitives.log", "v41native/qualification.json",
    "v41bridge/qualification.json", "isolation.json", "phase16-spirv.json",
    "two-pass/test.log", "two-pass/harness.log",
)
HEX64 = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout


def is_ancestor(ancestor, descendant):
    return subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
    ).returncode == 0


def _current_sha256(name):
    return digest(ROOT / name)


def _worst_logit(rows) -> float:
    values = [check["max_abs"] for row in rows for check in row["checks"]
              if "logits" in check.get("metric", "") and "max_abs" in check]
    require(values, "family evidence has no logit comparisons")
    return max(values)


def _verify_numerical_evidence(receipt_dir: Path) -> None:
    """Re-derive every retained numerical family from the shipped receipts."""
    from check_rdna3_gpt2_receipts import validate_qualification

    for name, count in (("shapes", 16), ("shared", 74), ("kimi", 3), ("seq2seq", 6),
                        ("gemma3", 4), ("llava", 4)):
        payload = json.loads((receipt_dir / "evidence" / name / "qualification.json").read_text())
        rows = validate_qualification(payload, count)
        require(_worst_logit(rows) < LIMIT, f"{name} evidence violates the logit gate")
    domains = [json.loads((receipt_dir / "evidence" / domain / "qualification.json").read_text())
               for domain in ("gemma4-legacy", "gemma4-rgb")]
    rows = [row for payload in domains for row in validate_qualification(payload, 1)]
    require(len({row["case"] for row in rows}) == 2,
            "gemma4 domains do not cover two distinct cases")
    require(_worst_logit(rows) < LIMIT, "gemma4 evidence violates the logit gate")
    native = json.loads((receipt_dir / "evidence" / "v41native" / "qualification.json").read_text())
    require(native.get("pass") is True and native.get("expected_tests") == 30
            and native.get("completed_tests") == 30,
            "V4.1 native evidence is incomplete")
    require(all(job.get("pass") for job in native.get("jobs", [])),
            "V4.1 native evidence reports a failing job")
    require(native.get("max_abs_logit", 1.0) < LIMIT,
            "V4.1 native evidence violates the logit gate")
    bridge = json.loads((receipt_dir / "evidence" / "v41bridge" / "qualification.json").read_text())
    require(bridge.get("pass") is True and bridge.get("expected_cases") == 2
            and bridge.get("completed_cases") == 2, "V4.1 bridge evidence is incomplete")
    isolation = json.loads((receipt_dir / "evidence" / "isolation.json").read_text())
    require(isolation.get("unchanged") is True and isolation.get("files_checked") == 1341
            and isolation.get("existing_recipes") == 672 and isolation.get("failures") == [],
            "isolation evidence changed")
    primitives = (receipt_dir / "evidence" / "primitives.log").read_text()
    require(re.search(r"test result: ok\. 9 passed; 0 failed;", primitives),
            "primitive evidence is incomplete")
    require(len(re.findall(r"^test .*rdna3_gpt2_.* \.\.\. ok$", primitives, re.M)) == 9,
            "primitive evidence coverage changed")


def _rederive(receipt_dir: Path) -> dict:
    """Recompute the graded two-pass summary from the shipped raw logs."""
    from emit_two_pass_evidence import parse

    parsed = parse(receipt_dir / "evidence" / "two-pass" / "test.log",
                   receipt_dir / "evidence" / "two-pass" / "harness.log")
    require(parsed["pass"], "shipped two-pass logs no longer re-derive a pass: "
            + "; ".join(parsed["failures"]))
    return dict(
        cases=len({row["case"] for row in parsed["cases"]}),
        comparisons=parsed["comparisons"],
        elements=parsed["elements"],
        bitwise_equal=parsed["bitwise_equal"],
        max_abs=parsed["max_abs"],
        max_ulp=parsed["max_ulp"],
        unwritten=parsed["unwritten"],
        dispatches=parsed["dispatches"],
        diagnostic=parsed["diagnostic"],
        adapters_identical=parsed["adapters"]["identical"],
        negative_controls=parsed["negative_controls"],
        test_log_sha256=parsed["evidence"]["test_log"],
        harness_log_sha256=parsed["evidence"]["harness_log"],
    )


def verify_receipt(receipt, receipt_dir, *, source_head, current_sha256=None,
                   ancestor_check=None, rederive=True, sealed_check=True):
    """Validate a Phase XVI receipt against the checkout and its own evidence."""
    current_sha256 = current_sha256 or _current_sha256
    ancestor_check = ancestor_check or is_ancestor
    require(isinstance(receipt, dict), "phase16 receipt is not an object")
    for key in REQUIRED_KEYS:
        require(key in receipt, "phase16 receipt is missing " + key)
    require(receipt["schema"] == 2, "phase16 receipt schema is not 2")
    require(receipt["scope"] == SCOPE, "phase16 receipt scope changed")
    require(receipt["target"] == "rdna3" and receipt["worktree"] is True,
            "phase16 receipt target or worktree flag changed")
    require(receipt["absolute_limit"] == LIMIT and receipt["rtol"] == 0
            and receipt["comparison"] == "strict_less_than",
            "phase16 acceptance policy changed")

    bound = receipt["source_head"]
    require(isinstance(bound, str) and len(bound) == 40, "phase16 receipt has no source revision")
    if bound != source_head:
        require(ancestor_check(bound, source_head),
                "phase16 receipt is bound to an unrelated revision")

    sources = receipt["sources"]
    require(isinstance(sources, dict) and len(sources) >= 134,
            "phase16 receipt binds fewer sources than the historical set")
    for name, expected in sources.items():
        require(current_sha256(name) == expected, "bound source changed: " + name)
    for name in EXTRA_SOURCES:
        require(name in sources, "phase16 receipt does not bind " + name)

    binaries = receipt["binaries"]
    for name in ("image_text_pipeline_probe", "gemma3_vision_probe", "gemma4_vision_probe",
                 "llava_pipeline_probe", "libtest"):
        value = binaries.get(name)
        require(isinstance(value, str) and len(value) == HEX64,
                "phase16 receipt has no usable " + name + " binding")

    shapes = receipt["shapes"]
    require(shapes.get("cases") == COUNTS["shapes"] and shapes.get("worst_logit", 1.0) < LIMIT,
            "phase16 shape family no longer covers 16 cases under the strict limit")
    retained = receipt["retained"]
    for name in GATES:
        row = retained.get(name, {})
        require(row.get("cases") == COUNTS[name] and row.get("pass_") is True
                and row.get("worst_logit", 1.0) < LIMIT,
                f"phase16 retained gate {name} is incomplete or over the limit")
    primitives = receipt["primitives"]
    require(primitives.get("passed") == 9 and primitives.get("failed") == 0,
            "phase16 GPT-2 primitives are incomplete")
    native = receipt["v41"].get("native", {})
    require(native.get("tests") == 30 and native.get("max_abs_logit", 1.0) < LIMIT,
            "phase16 V4.1 native gate is incomplete")
    require(receipt["v41"].get("bridge", {}).get("cases") == 2,
            "phase16 V4.1 bridge gate is incomplete")
    isolation = receipt["isolation"]
    require(isolation.get("files_checked") == 1341 and isolation.get("existing_recipes") == 672,
            "phase16 isolation audit changed shape")

    shaders = receipt["shaders"]
    require(shaders.get("inventory") == 45 and shaders.get("status_counts") == {"identical": 45},
            "phase16 production inventory is not 45 identical artifacts")
    historical = shaders.get("historical", {})
    require(historical.get("artifacts") == 43 and historical.get("verified") == 43
            and historical.get("recompiled") == 0,
            "phase16 historical shader scope changed")
    sealed_rows = historical.get("rows") or []
    require(len(sealed_rows) == 43, "phase16 receipt does not list the 43 sealed artifacts")
    if sealed_check:
        sealed = json.loads(SEALED_GATE.read_text())
        require(digest(SEALED_GATE) == historical.get("sealed_gate_sha256"),
                "sealed Phase XV shader gate changed")
        index = {row["artifact"]: row for row in sealed["artifacts"]}
        for row in sealed_rows:
            name = row.get("artifact")
            require(name in index, "historical artifact not in the sealed gate: " + str(name))
            require(row.get("sha256") == index[name]["sha256"],
                    "historical artifact digest changed: " + str(name))
            require(digest(SHADERS / name) == row["sha256"],
                    "committed historical artifact changed: " + name)
    phase16 = shaders.get("phase16", {})
    require(phase16.get("artifacts") == 2 and phase16.get("rebuilt") == 2
            and phase16.get("validated") == 2,
            "phase16 rebuilt shader scope changed")
    new_rows = phase16.get("rows") or []
    require({row.get("artifact") for row in new_rows} == set(PHASE16_ARTIFACTS),
            "phase16 receipt does not describe exactly the two new artifacts")
    for row in new_rows:
        name = row["artifact"]
        require(row.get("rebuilt_sha256") == row.get("sha256"),
                f"{name}: rebuilt bytes were not byte-identical")
        require(row.get("validated") is True, f"{name}: SPIR-V validation is missing")
        require(digest(SHADERS / name) == row["sha256"], f"{name}: committed artifact changed")

    backward = receipt["attention_backward"]
    require(backward.get("enabled_by") == ENVIRONMENT and backward.get("scalar_default") is True,
            "phase16 two-pass opt-in contract changed")
    require(backward.get("cases") == 4 and backward.get("comparisons") == 12,
            "phase16 gradient coverage is not four cases and twelve tensors")
    require(backward.get("elements", 0) > 0
            and backward.get("bitwise_equal") == backward.get("elements"),
            "phase16 gradient comparison is not bit-exact")
    require(backward.get("max_abs") == 0.0 and backward.get("max_ulp") == 0
            and backward.get("unwritten") == 0,
            "phase16 gradient evidence reports an error or an unwritten element")
    require(backward.get("adapters_identical") is True,
            "phase16 adapter outputs are not bit-identical")
    dispatches = backward.get("dispatches") or {}
    control = dispatches.get("scalar_arm") or {}
    candidate = dispatches.get("twopass_arm") or {}
    require(control.get("scalar", 0) > 0 and control.get("query") == 0 and control.get("key") == 0,
            "phase16 scalar control arm did not run the scalar backward alone")
    require(candidate.get("scalar") == 0 and candidate.get("query", 0) > 0
            and candidate.get("key", 0) > 0,
            "phase16 candidate arm did not dispatch both passes without falling back")
    require("kernels=true" in str(backward.get("diagnostic", ""))
            and "unfused_qk=9" in str(backward.get("diagnostic", "")),
            "phase16 mode-9 dispatch diagnostic is missing")

    evidence = receipt["evidence"]
    require(isinstance(evidence, dict) and evidence, "phase16 receipt binds no evidence")
    for name in REQUIRED_EVIDENCE:
        require(name in evidence, "phase16 receipt does not bind evidence " + name)
    for name, expected in evidence.items():
        path = receipt_dir / "evidence" / name
        require(path.is_file(), "phase16 evidence file is missing: " + name)
        require(digest(path) == expected, "phase16 evidence file changed: " + name)

    _verify_numerical_evidence(receipt_dir)
    if rederive:
        derived = _rederive(receipt_dir)
        for key, value in derived.items():
            require(backward.get(key) == value,
                    "phase16 two-pass evidence does not re-derive: " + key)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=CURRENT_RECEIPT)
    parser.add_argument("--source-head")
    args = parser.parse_args()
    head = args.source_head or git("rev-parse", "HEAD").strip()
    require(len(head) == 40, "cannot resolve the checkout revision")
    if not args.receipt.is_file():
        print(f"PHASE XVI CURRENT SOURCE: UNQUALIFIED (no {args.receipt} for {head[:12]})")
        return 3
    receipt = json.loads(args.receipt.read_text())
    try:
        verify_receipt(receipt, args.receipt.parent, source_head=head)
    except ValueError as error:
        print(f"PHASE XVI CURRENT SOURCE: FAIL ({error})")
        return 1
    print(f"PHASE XVI CURRENT SOURCE: PASS ({head[:12]}, "
          f"{receipt['attention_backward']['comparisons']} gradient tensors, "
          f"{receipt['shaders']['inventory']} shader artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fail closed on the RDNA3 multimodal receipts.

This is the anti-overclaim guard. It binds every retained receipt by SHA-256,
re-derives the inventory totals from the per-profile rows instead of trusting
the summary fields, and refuses any document that claims 44/44 green or a
qualified RDNA3 multimodal inventory without the GPT-2-wide closure evidence.

The acceptance policy is read from the receipts and must be the unchanged
``max_abs < 2e-7`` / ``rtol = 0`` strict comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
RECEIPTS = (
    "release_parity/multimodal_training/20261106-rdna3-profile-completion"
)
EXPECTED_PROFILES = 44
EXPECTED_GREEN = 44
RED_PROFILE = "gpt2_wide"
RED_CASE = "gpt2_wide/unmasked"
EXPECTED_CASES = 85
EXPECTED_PASSING = 85
SIX_PROFILES = (
    "bart",
    "t5",
    "switch_transformers",
    "kimi_k3_kda",
    "kimi_k3_mla",
    "kimi_k3_text",
)
LIMIT = 2e-7
MANIFEST = "receipt-hashes.json"
BOUND_FILES = (
    f"{RECEIPTS}/inventory-status.json",
    f"{RECEIPTS}/six-profile-replay.json",
    f"{RECEIPTS}/gpt2-wide-residual.json",
    f"{RECEIPTS}/seq2seq-six-qualification.json",
    f"{RECEIPTS}/kimi-three-qualification.json",
    f"{RECEIPTS}/v41-native-full-qualification.json",
    f"{RECEIPTS}/v41-shared-bridge-qualification.json",
    f"{RECEIPTS}/v41-shared-image-profile.json",
    "intel-handoff/multimodal-gen9-expected-values.json",
    "intel-handoff/multimodal-gen9-status.json",
)
DOCUMENTS = (
    "VALIDATION_STATUS.md",
    "RDNA3_MULTIMODAL_PORT.md",
    "RDNA3_MULTIMODAL_HANDOFF.md",
    "hierarchos-vulkan/validation/intel-handoff/MULTIMODAL_GEN9.md",
    f"hierarchos-vulkan/validation/{RECEIPTS}/README.md",
)
CLAIM = re.compile(r"(?<![\d.])44\s*/\s*44")
NEGATION = re.compile(
    r"\b(not|no|never|cannot|can't|isn't|is not|without|unset)\b", re.IGNORECASE
)


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_claims(node: Any, where: str, hits: list[tuple[str, str]]) -> None:
    """Collect inventory-scope claims so a nested receipt cannot slip through."""
    if isinstance(node, dict):
        if "fully_green_profiles" in node:
            hits.append((where, f"fully_green_profiles={node['fully_green_profiles']}"))
        for key in (
            "rdna3_multimodal_qualified",
            "all_profiles_green",
            "all_green_backbones_qualified",
        ):
            if node.get(key) is True:
                hits.append((where, f"{key}=true"))
        for key, value in node.items():
            walk_claims(value, f"{where}.{key}", hits)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk_claims(value, f"{where}[{index}]", hits)


def paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)
    return [re.sub(r"\s+", " ", block) for block in blocks if block.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=VALIDATION / RECEIPTS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    receipts = args.receipts.resolve()
    manifest_path = (args.manifest or receipts / MANIFEST).resolve()
    errors: list[str] = []

    if args.write_manifest:
        manifest = {
            "schema": 1,
            "files": {
                name: digest(VALIDATION / name)
                for name in BOUND_FILES
                if (VALIDATION / name).is_file()
            },
        }
        missing = [name for name in BOUND_FILES if name not in manifest["files"]]
        if missing:
            print("FAIL: cannot bind missing receipt(s):", ", ".join(missing))
            return 1
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"[receipt-guard] wrote {manifest_path.name} ({len(manifest['files'])} files)")

    try:
        manifest = load(manifest_path)
    except FileNotFoundError:
        print(f"FAIL: missing {manifest_path}; run with --write-manifest")
        return 1
    bound = manifest.get("files", {})
    missing = sorted(set(BOUND_FILES) - set(bound))
    extra = sorted(set(bound) - set(BOUND_FILES))
    if missing or extra:
        errors.append(f"receipt manifest set mismatch; missing={missing} extra={extra}")
    for name, expected in sorted(bound.items()):
        path = VALIDATION / name
        if not path.is_file():
            errors.append(f"missing bound receipt: {name}")
            continue
        actual = digest(path)
        if actual != expected:
            errors.append(f"receipt hash mismatch: {name} ({actual} != {expected})")

    inventory = load(receipts / "inventory-status.json")
    six = load(receipts / "six-profile-replay.json")
    residual = load(receipts / "gpt2-wide-residual.json")
    v41 = load(receipts / "v41-native-full-qualification.json")
    bridge = load(receipts / "v41-shared-bridge-qualification.json")
    seq2seq = load(receipts / "seq2seq-six-qualification.json")
    kimi = load(receipts / "kimi-three-qualification.json")
    gen9 = load(VALIDATION / "intel-handoff/multimodal-gen9-status.json")

    acceptance = inventory.get("acceptance", {})
    if (
        acceptance.get("absolute_limit") != LIMIT
        or acceptance.get("rtol") != 0
        or acceptance.get("comparison") != "strict_less_than"
    ):
        errors.append(f"inventory acceptance changed: {acceptance}")

    profiles = inventory.get("profiles", {})
    if len(profiles) != EXPECTED_PROFILES:
        errors.append(f"expected {EXPECTED_PROFILES} profile rows, found {len(profiles)}")
    derived_failing = sorted(name for name, row in profiles.items() if not row.get("pass"))
    derived_green = len(profiles) - len(derived_failing)
    if derived_failing:
        errors.append(f"red profiles are {derived_failing}, expected none")
    if derived_green != EXPECTED_GREEN:
        errors.append(f"derived green profiles {derived_green}, expected {EXPECTED_GREEN}")
    summary = inventory.get("inventory", {})
    if summary.get("fully_green_profiles") != derived_green:
        errors.append(
            f"summary fully_green_profiles={summary.get('fully_green_profiles')} "
            f"but profile rows derive {derived_green}"
        )
    if summary.get("failing_profiles") != derived_failing:
        errors.append(
            f"summary failing_profiles={summary.get('failing_profiles')} "
            f"but profile rows derive {derived_failing}"
        )
    expected_cases = sum(row.get("expected_cases", 0) for row in profiles.values())
    completed_cases = sum(row.get("completed_cases", 0) for row in profiles.values())
    derived_passing = sum(
        row.get("completed_cases", 0) - len(row.get("failing_cases", []))
        for row in profiles.values()
    )
    if (expected_cases, completed_cases, derived_passing) != (
        EXPECTED_CASES,
        EXPECTED_CASES,
        EXPECTED_PASSING,
    ):
        errors.append(
            "case totals derive "
            f"{completed_cases}/{expected_cases} completed, {derived_passing} passing; "
            f"expected {EXPECTED_CASES}/{EXPECTED_CASES}, {EXPECTED_PASSING}"
        )
    cases = inventory.get("cases", {})
    for key, value in (
        ("expected", EXPECTED_CASES),
        ("completed", EXPECTED_CASES),
        ("passing", EXPECTED_PASSING),
        ("failing", 0),
    ):
        if cases.get(key) != value:
            errors.append(f"case count {key}={cases.get(key)}, expected {value}")
    for key in (
        "all_profiles_green",
        "all_green_backbones_qualified",
        "rdna3_multimodal_qualified",
    ):
        if summary.get(key) is not True:
            errors.append(f"inventory must record {key}=true, found {summary.get(key)}")

    gpt2 = profiles.get(RED_PROFILE, {})
    if gpt2.get("pass") is not True:
        errors.append("gpt2_wide must be green")
    if gpt2.get("failing_cases") != []:
        errors.append(f"gpt2_wide failing_cases={gpt2.get('failing_cases')}")
    if not isinstance(gpt2.get("worst_logit"), (int, float)) or (
        gpt2["worst_logit"] >= LIMIT
    ):
        errors.append(f"gpt2_wide worst_logit={gpt2.get('worst_logit')} is not below the gate")

    if six.get("all_pass") is not True:
        errors.append("six-profile replay is not all green")
    if six.get("profile_count") != len(SIX_PROFILES) or set(six.get("profiles", {})) != set(
        SIX_PROFILES
    ):
        errors.append("six-profile replay does not cover exactly the six profiles")
    six_cases = sum(row.get("completed_cases", 0) for row in six.get("profiles", {}).values())
    if six.get("case_count") != six_cases:
        errors.append(f"six-profile case_count={six.get('case_count')} but rows total {six_cases}")
    for name in SIX_PROFILES:
        if profiles.get(name, {}).get("pass") is not True:
            errors.append(f"inventory does not mark {name} green")
    if seq2seq.get("pass") is not True or seq2seq.get("completed_cases") != 6:
        errors.append("seq2seq receipt is not a complete 6/6 pass")
    if kimi.get("pass") is not True or kimi.get("completed_cases") != 3:
        errors.append("KIMI receipt is not a complete 3/3 pass")

    if v41.get("pass") is not True or v41.get("profile_qualified") is not True:
        errors.append("native V4.1 receipt is not a qualified pass")
    if v41.get("completed_tests") != v41.get("expected_tests"):
        errors.append("native V4.1 did not complete every test")
    if v41.get("max_abs_logit") != 0.0:
        errors.append(f"native V4.1 max_abs_logit={v41.get('max_abs_logit')}, expected 0.0")
    if profiles.get("deepseek_v41_scalable", {}).get("pass") is not True:
        errors.append("inventory does not mark deepseek_v41_scalable green")
    if bridge.get("pass") is not True or bridge.get("completed_cases") != 2:
        errors.append("V4.1 shared-image bridge is not a complete 2/2 pass")

    if residual.get("acceptance", {}).get("absolute_limit") != LIMIT:
        errors.append("residual receipt does not use the 2e-7 gate")
    conclusion = residual.get("conclusion", {})
    if conclusion.get("profile_gate") != "PASS" or conclusion.get("waived") is not False:
        errors.append("residual receipt must record a non-waived PASS")
    if conclusion.get("closure_status") != "closed":
        errors.append("residual receipt must record closure_status=closed")
    if conclusion.get("forward_exact") is not True:
        errors.append("residual receipt must record forward_exact=true")
    if residual.get("forward_oracle", {}).get("all_exact") is not True:
        errors.append("residual receipt forward oracle is not exact")
    if residual.get("gate", {}).get("pass") is not True:
        errors.append("residual receipt gate is not a pass")
    if gen9.get("physical_run") is not False:
        errors.append("Gen9 status must not claim a physical run on this host")
    if gen9.get("receipt_issued") is not False or gen9.get("qualified") is not False:
        errors.append("Gen9 status must not issue a qualification receipt")
    if gen9.get("status") != "blocked":
        errors.append(f"unexpected Gen9 status {gen9.get('status')}")
    adapters = gen9.get("host", {}).get("adapters", [])
    if any(
        str(adapter.get("vendor_id", "")).lower() == "0x8086"
        and str(adapter.get("device_id", "")).lower() == "0x1916"
        for adapter in adapters
    ):
        errors.append("Gen9 record lists an Intel 8086:1916 adapter but claims no run")

    if errors:
        print("FAIL")
        for error in errors:
            print(f" - {error}")
        return 1

    print(f"[receipt-guard] sha256 verified: {len(bound)} receipts")
    print(
        f"[receipt-guard] inventory {derived_green}/{len(profiles)} green, "
        f"{derived_passing}/{completed_cases} cases pass, all profiles green"
    )
    print(f"[receipt-guard] six-profile replay: {six_cases}/{six_cases} cases pass")
    print(
        f"[receipt-guard] native V4.1: {v41['completed_tests']}/"
        f"{v41['expected_tests']} tests, max_abs_logit={v41['max_abs_logit']}"
    )
    print("[receipt-guard] Gen9 physical run recorded as blocked, no receipt issued")
    print("[receipt-guard] 44/44 and qualified claims are evidence-backed")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

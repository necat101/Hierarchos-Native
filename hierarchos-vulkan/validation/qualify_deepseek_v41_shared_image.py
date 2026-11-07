"""Bridge frozen native DeepSeek V4.1 image training into the shared inventory.

V4.1 is not a shared CLIP text backbone. This module therefore does not route
it through ``qualify_shared_image_pipeline``'s CLIP adapter. It binds the two
cases in the maintained native joint fixture to the same 44-profile inventory
and replays the existing ``DeepseekV41MultimodalTrainer`` libtest. The receipt
uses the shared acceptance policy and can be consumed by the inventory summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from qualify_multimodal_training import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
INVENTORY = VALIDATION / "multimodal_training_inventory.json"
SOURCE_REL = Path(
    "hierarchos-vulkan/validation/fixtures/"
    "deepseek_v41_native_pipeline_avx2.json"
)
SOURCE = ROOT / SOURCE_REL
PROFILE = "deepseek_v41_scalable"
SOURCE_SHA256 = (
    "9f5592989820c1567caf3fd4ca6b5354f7bb8e78dfb6b38f1ebb43b9b4cc403b"
)
TEST = "deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu"
LIMIT = 2e-7
EXPECTED_CASES = 2
EXPECTED_CASE_NAMES = {"native_vit_4x4", "native_vit_6x4"}
EXPECTED_LOGITS = 1360
EXPECTED_COMPARISONS = 1866
EXPECTED_COMPARISON_VALUES = 155444
LOG_RE = re.compile(
    r"^V4\.1 full training ([^\r\n]+/(?:inference|training)-logits): "
    r"(\d+) values, worst (\S+) at",
    re.M,
)


def required_profiles() -> list[str]:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return [entry["profile"] for entry in data["profiles"]]


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def source_cases() -> tuple[dict, list[dict]]:
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("frozen native V4.1 image fixture changed")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    names = {case.get("name") for case in cases}
    if len(cases) != EXPECTED_CASES or names != EXPECTED_CASE_NAMES:
        raise ValueError("native V4.1 image fixture cases changed")
    return data, cases


def prepare(args) -> None:
    requested = args.profiles or [PROFILE]
    if not requested or set(requested) != {PROFILE}:
        raise ValueError("this bridge prepares only deepseek_v41_scalable")
    data, cases = source_cases()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    entries = []
    for case in cases:
        image = case["image"]
        entries.append(
            {
                "name": case["name"],
                "profile": PROFILE,
                "source_case_sha256": hashlib.sha256(
                    canonical(case).encode("utf-8")
                ).hexdigest(),
                "image": {
                    key: image.get(key)
                    for key in (
                        "width",
                        "height",
                        "n_vit_h",
                        "n_vit_w",
                        "n_llm_h",
                        "n_llm_w",
                        "png_sha256",
                    )
                },
                "batch_size": case.get("batch_size"),
                "seq_len": case.get("seq_len"),
                "adamw_updates": 2,
            }
        )

    manifest = {
        "schema": 1,
        "complete": True,
        "scope": "shared_inventory_native_deepseek_v41_image_training",
        "profile": PROFILE,
        "architecture": "deepseek_v41",
        "bridge": "native_v41_vision_trainer",
        "native_contract": data.get("contract"),
        "absolute_limit": LIMIT,
        "rtol": 0,
        "comparison": "strict_less_than",
        "required_green_profiles": required_profiles(),
        "requested_profiles": [PROFILE],
        "pending": [],
        "source_fixture": SOURCE_REL.as_posix(),
        "source_fixture_sha256": SOURCE_SHA256,
        "cases": entries,
        "oracle": {
            "cpu": data.get("cpu"),
            "threads": data.get("threads"),
            "torch": data.get("torch"),
            "torch_revision": data.get("torch_revision"),
            "revision": data.get("revision"),
            "component_fixture_sha256": data.get("component_fixture_sha256"),
            "source_sha256": data.get("source_sha256", {}),
        },
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "profile.json",
        {
            "schema": 1,
            "profile": PROFILE,
            "architecture": "deepseek_v41",
            "bridge": "native_v41_vision_trainer",
            "cases": [entry["name"] for entry in entries],
            "source_fixture_sha256": SOURCE_SHA256,
        },
    )
    print(
        f"Prepared {PROFILE}: {len(entries)} native image cases at "
        f"{SOURCE_REL.as_posix()}"
    )


def validate_native(native: dict) -> None:
    if native.get("pass") is not True:
        raise ValueError("native V4.1 joint receipt is not green")
    if native.get("expected_cases") != EXPECTED_CASES:
        raise ValueError("native V4.1 expected case count changed")
    if native.get("completed_cases") != EXPECTED_CASES:
        raise ValueError("native V4.1 did not complete both cases")
    if native.get("optimizer_updates") != 4:
        raise ValueError("native V4.1 optimizer update count changed")
    if native.get("logit_values") != EXPECTED_LOGITS:
        raise ValueError("native V4.1 logit count changed")
    if native.get("failures") != []:
        raise ValueError("native V4.1 joint receipt contains failures")
    comparisons = native.get("all_comparisons", {})
    if (
        comparisons.get("comparisons") != EXPECTED_COMPARISONS
        or comparisons.get("values") != EXPECTED_COMPARISON_VALUES
    ):
        raise ValueError("native V4.1 joint comparison coverage changed")


def run(args) -> None:
    root = args.fixtures.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("profile") != PROFILE
        or manifest.get("absolute_limit") != LIMIT
        or manifest.get("rtol") != 0
        or manifest.get("comparison") != "strict_less_than"
    ):
        raise ValueError("altered or incomplete native V4.1 profile fixture")
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("frozen native V4.1 image fixture changed")
    cases = manifest.get("cases", [])
    if len(cases) != EXPECTED_CASES or {c["name"] for c in cases} != EXPECTED_CASE_NAMES:
        raise ValueError("native V4.1 profile fixture cases changed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    binary_sha = digest(binary)
    joint_path = output / "native-joint.json"
    env = dict(
        os.environ,
        HIERARCHOS_V41_JOINT_QUALIFICATION=str(joint_path),
    )
    process = subprocess.run(
        [str(binary), TEST, "--test-threads=1", "--nocapture"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log = process.stdout + process.stderr
    (output / "run.log").write_text(log, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"native V4.1 joint test failed with {process.returncode}")
    if digest(binary) != binary_sha:
        raise ValueError("native V4.1 test binary changed during replay")
    if not joint_path.exists():
        raise ValueError("native V4.1 joint receipt was not written")
    native = json.loads(joint_path.read_text(encoding="utf-8"))
    validate_native(native)

    parsed = []
    for match in LOG_RE.finditer(log):
        label, values, worst = match.groups()
        parsed.append(
            {
                "case": label.split("/", 1)[0],
                "phase": label.split("/", 1)[1],
                "values": int(values),
                "max_abs": float(worst),
            }
        )
    if (
        len(parsed) != 10
        or len({(row["case"], row["phase"]) for row in parsed}) != 10
        or sum(row["values"] for row in parsed) != EXPECTED_LOGITS
        or any(
            not math.isfinite(row["max_abs"]) or row["max_abs"] >= LIMIT
            for row in parsed
        )
    ):
        raise ValueError("native V4.1 logit coverage or strict limit changed")

    results = []
    for case in cases:
        rows = [row for row in parsed if row["case"] == case["name"]]
        if not rows:
            raise ValueError(f"no native V4.1 logit metrics for {case['name']}")
        checks = [
            {
                "metric": f"{case['name']}/{row['phase']}",
                "values": row["values"],
                "max_abs": row["max_abs"],
                "pass": row["max_abs"] < LIMIT,
            }
            for row in rows
        ]
        results.append(
            {
                "case": case["name"],
                "profile": PROFILE,
                "pass": all(check["pass"] for check in checks),
                "checks": checks,
            }
        )

    max_abs = max(row["max_abs"] for row in parsed)
    passed = all(result["pass"] for result in results)
    passed = bool(native.get("pass") is True and passed)
    receipt = {
        "schema": 1,
        "scope": manifest["scope"],
        "profile": PROFILE,
        "architecture": "deepseek_v41",
        "bridge": manifest["bridge"],
        "target": args.target,
        "device_id": args.device_id,
        "device_selection": "local default Vulkan adapter; target supplied by the operator",
        "absolute_limit": LIMIT,
        "rtol": 0,
        "comparison": "strict_less_than",
        "expected_cases": EXPECTED_CASES,
        "completed_cases": len(results),
        "complete": len(results) == EXPECTED_CASES,
        "pass": passed,
        "profile_qualified": passed,
        "results": results,
        "native_cases": native["completed_cases"],
        "optimizer_updates": native["optimizer_updates"],
        "logit_values": native["logit_values"],
        "max_abs_logit": max_abs,
        "joint_comparisons": native.get("all_comparisons"),
        "native_joint_pass": native.get("pass") is True,
        "native_joint_sha256": digest(joint_path),
        "fixture_manifest_sha256": digest(manifest_path),
        "source_fixture": manifest["source_fixture"],
        "source_fixture_sha256": SOURCE_SHA256,
        "binary_sha256": binary_sha,
        "runner_sha256": digest(Path(__file__).resolve()),
        "aten_vector_width": 8,
        "limits": [
            "one complete image span per prefill",
            "fixed native patch geometry per trainer",
            "production geometry and BF16 training unqualified",
            (
                "physical RDNA3 replay recorded on the local default Vulkan adapter"
                if args.target == "rdna3"
                else "physical RDNA3 replay pending"
            ),
        ],
    }
    write_json(output / "qualification.json", receipt)
    if not passed:
        raise SystemExit(1)
    print(
        f"DeepSeek V4.1 shared-image bridge PASS: {len(results)}/{EXPECTED_CASES} "
        f"cases, {len(parsed)} logit checks, max_abs {max_abs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profiles", nargs="+")
    p = sub.add_parser("run")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", choices=["gen9", "rdna3"], required=True)
    p.add_argument("--device-id", type=lambda x: int(x, 0))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()

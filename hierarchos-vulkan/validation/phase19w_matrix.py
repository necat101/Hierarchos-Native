"""Build and validate the Phase XIX-W per-architecture compatibility matrix.

The matrix is source-bound to the Rust registry and the retained receipts.  It
never infers numerical qualification from registry membership: a canonical
architecture is historical-only when it appears in a sealed receipt and
`historical_pass_preserved` only when every current replay row for it passes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from phase19w_registry import (
    ALIAS_COUNT,
    CANONICAL_COUNT,
    ROOT,
    TOTAL_COUNT,
    digest,
    parse_registry,
    read_json,
)

POLICY = {
    "absolute_limit": 2e-7,
    "rtol": 0,
    "comparison": "strict_less_than",
}
STATUSES = {
    "historical_pass_preserved",
    "newly_qualified",
    "functional_smoke_only",
    "unsupported_capability",
    "not_tested",
    "regression",
    "blocked",
}
DEFAULT_HISTORICAL = (
    ROOT
    / "hierarchos-vulkan/validation/release_parity/gen9_architectures/20261101/qualification.json"
)
DEFAULT_PEFT = (
    ROOT
    / "hierarchos-vulkan/validation/release_parity/parameter_streaming/qualification.json"
)
DEFAULT_OUTPUT = ROOT / "hierarchos-vulkan/validation/architecture_compatibility/phase19w-matrix.json"

# These fixture profiles are text PEFT profiles whose declared native graph is
# explicit in the existing qualification harness.  They are only used to add a
# training capability when the retained PEFT receipt really contains the row.
PEFT_PROFILE_TARGETS = {
    "phi4_multimodal_text": "phi3",
    "kimi_k25_text": "deepseek_v3",
    "kimi_k3_kda": "kimi_linear",
    "kimi_k3_mla": "kimi_linear",
    "kimi_k3_text": "kimi_linear",
    "qwen2_5_gqa": "qwen2",
    "qwen2_5_sliding_tied": "qwen2",
    "qwen3_5_full": "qwen3_5_text",
    "qwen3_5_linear": "qwen3_5_text",
    "qwen3_5_mixed": "qwen3_5_text",
    "qwen3_5_moe": "qwen3_5_moe_text",
    "qwen4_exp_linear": "qwen4_exp_text",
    "qwen4_exp_qsa": "qwen4_exp_text",
    "qwen4_exp_ple": "qwen4_exp_text",
    "qwen4_exp_mixed_ple": "qwen4_exp_text",
    "minimax_m3_dense": "minimax_m3_vl_text",
    "minimax_m3": "minimax_m3_vl_text",
    "gemma4": "gemma4_text",
    "gemma3": "gemma3_text",
    "falcon_h1_gated_after": "falcon_h1",
    "falcon_h1_gated_before": "falcon_h1",
    "falcon_h1_silu": "falcon_h1",
}


def sha(path: Path) -> str:
    return digest(Path(path))


def _source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _forward_rows(path: Path) -> list[dict]:
    value = read_json(path)
    records = value.get("records", [])
    if not records:
        raise ValueError(f"no forward records in {path}")
    return records


def _forward_by_architecture(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        architecture = record.get("architecture")
        if not architecture or record.get("mode") not in {"resident", "streaming"}:
            raise ValueError("forward record lacks architecture or execution mode")
        grouped.setdefault(architecture, []).append(record)
    return grouped


def _generation_by_architecture(
    path: Path | None, family_to_architecture: dict[str, str] | None = None
) -> tuple[dict[str, list[dict]], dict]:
    if path is None or not Path(path).is_file():
        return {}, {}
    value = read_json(Path(path))
    grouped: dict[str, list[dict]] = {}
    for record in value.get("records", []):
        architecture = record.get("architecture") or (family_to_architecture or {}).get(
            record.get("family")
        )
        if not architecture:
            raise ValueError("generation record has no architecture mapping")
        grouped.setdefault(architecture, []).append(record)
    return grouped, value


def _peft_by_architecture(path: Path) -> tuple[dict[str, dict], dict]:
    value = read_json(path)
    profiles = value.get("profiles", {})
    grouped: dict[str, dict] = {}
    for profile, row in profiles.items():
        target = PEFT_PROFILE_TARGETS.get(profile)
        if target:
            grouped[target] = {**row, "profile": profile}
    return grouped, value


def _alias_target(target: str | None) -> str:
    return target if target and target != "None" else "nested_text_config"


def build(
    forward_path: Path,
    historical_path: Path,
    peft_path: Path,
    generation_path: Path | None,
    output: Path,
) -> dict:
    registry = parse_registry()
    historical = read_json(historical_path)
    forward = read_json(forward_path)
    peft, peft_value = _peft_by_architecture(peft_path)
    records = _forward_rows(forward_path)
    by_architecture = _forward_by_architecture(records)
    family_to_architecture = {
        row["family"]: row["architecture"] for row in records
    }
    generation, generation_value = _generation_by_architecture(
        generation_path, family_to_architecture
    )
    historical_native = set(historical.get("forward", {}).get("native_architectures", []))
    if len(historical_native) != 26:
        raise ValueError("historical receipt does not identify 26 native architectures")
    if (forward.get("absolute_limit"), forward.get("rtol"), forward.get("comparison")) != (
        POLICY["absolute_limit"],
        POLICY["rtol"],
        POLICY["comparison"],
    ):
        raise ValueError("current forward policy differs from Phase XIX-W")
    if not forward.get("pass") or len(records) != 166:
        raise ValueError("current 166-row forward replay is not green")
    if any(not row.get("pass") for row in records):
        raise ValueError("current forward replay contains a failing row")
    if any(
        row.get("vendor_id") != 0x1002 or row.get("device_id") != 0x15BF
        for row in records
    ):
        raise ValueError("current forward replay is not from physical RDNA3 1002:15bf")

    canonical_arch = set(registry["canonical"])
    aliases = registry["aliases"]
    historical_training_profiles = set(historical.get("rdna_handoff_profiles", {}).get("profiles", []))
    peft_profile_names = set(peft_value.get("profiles", {}))
    if historical_training_profiles - peft_profile_names:
        raise ValueError("a historical training profile is absent from the PEFT receipt")
    generation_rows = sum(len(rows) for rows in generation.values())
    generation_ok = bool(generation_value.get("pass")) and generation_rows == 72
    if generation_value and not generation_ok:
        raise ValueError("generation replay is not the complete green 72-request matrix")

    matrix_records = []
    for architecture in registry["canonical"]:
        rows = by_architecture.get(architecture, [])
        cases = sorted({row["family"] for row in rows})
        modes = sorted({row["mode"] for row in rows})
        passed = bool(rows) and all(row.get("pass") is True for row in rows)
        if architecture in historical_native:
            if not rows:
                status = "blocked"
            else:
                status = "historical_pass_preserved" if passed else "regression"
        elif rows:
            status = "functional_smoke_only" if passed else "regression"
        else:
            status = "not_tested"
        training = architecture in peft
        training_capability = {
            "profile": peft[architecture]["profile"],
            "max_abs": peft[architecture].get("max_abs"),
            "resident_and_streamed": True,
        } if training else None
        gen_rows = generation.get(architecture, [])
        generation_capability = {
            "runs": len(gen_rows),
            "pass": bool(gen_rows) and all(row.get("pass") is True for row in gen_rows),
        } if gen_rows else None
        capabilities = [
            "config_parse",
            "safetensors_load",
            "native_vulkan_forward",
            "native_package_export",
        ]
        if generation_capability:
            capabilities.append("generation")
        if training:
            capabilities.append("lora_training")
            capabilities.append("backward")
        matrix_records.append(
            {
                "model_type": architecture,
                "canonical_model_type": architecture,
                "is_alias": False,
                "architecture_family": architecture,
                "declared_capabilities": capabilities,
                "historically_qualified_capabilities": [
                    capability
                    for capability, present in [
                        ("forward", architecture in historical_native),
                        ("generation", architecture in generation),
                        ("training", training),
                    ]
                    if present
                ],
                "historical_receipt_revision": (
                    historical.get("base_revision") if architecture in historical_native else None
                ),
                "current_source_revision": forward.get("source_revision") or _source_revision(),
                "tested_execution_modes": modes,
                "device": {
                    "name": "AMD Radeon Graphics / RDNA3",
                    "vendor_id": 0x1002,
                    "device_id": 0x15BF,
                } if rows else None,
                "fixture_identity": {
                    "manifest_sha256": forward.get("manifest_sha256"),
                    "profiles": cases,
                } if rows else None,
                "forward_result": {
                    "pass": passed,
                    "runs": len(rows),
                    "max_abs": max((row.get("max_abs", 0.0) for row in rows), default=None),
                } if rows else None,
                "backward_result": training_capability,
                "training_result": training_capability,
                "generation_result": generation_capability,
                "export_reload_result": None,
                "fallback_behavior": "none_declared",
                "numerical_policy": POLICY,
                "execution_evidence": (
                    f"gen9_architectures/20261101/qualification.json + {forward_path.name}"
                    if rows
                    else "none"
                ),
                "overall_status": status,
            }
        )

    alias_records = []
    for alias in aliases:
        target = _alias_target(registry["alias_targets"].get(alias))
        alias_records.append(
            {
                "model_type": alias,
                "canonical_model_type": target,
                "is_alias": True,
                "architecture_family": "nested-text-config" if target == "nested_text_config" else target,
                "declared_capabilities": ["config_parse", "text_backbone_resolution"],
                "historically_qualified_capabilities": [],
                "historical_receipt_revision": None,
                "current_source_revision": _source_revision(),
                "tested_execution_modes": [],
                "device": None,
                "fixture_identity": None,
                "forward_result": None,
                "backward_result": None,
                "training_result": None,
                "generation_result": None,
                "export_reload_result": None,
                "fallback_behavior": "auxiliary_tensors_preserved_where_supported",
                "numerical_policy": POLICY,
                "execution_evidence": "source_alias_route_verified; no numerical alias oracle",
                "alias_resolution_result": "nested_text_config" if target == "nested_text_config" else "source_route_verified",
                "overall_status": "not_tested",
            }
        )

    all_records = matrix_records + alias_records
    if len(all_records) != TOTAL_COUNT:
        raise ValueError(f"matrix has {len(all_records)} rows, expected {TOTAL_COUNT}")
    if {row["model_type"] for row in alias_records} != set(aliases):
        raise ValueError("matrix does not account for every alias exactly once")
    if {row["model_type"] for row in matrix_records} != canonical_arch:
        raise ValueError("matrix does not account for every canonical architecture exactly once")
    if not generation_ok:
        raise ValueError("generation evidence is required for the Phase XIX-W matrix")
    summary = {
        "canonical_types": len(matrix_records),
        "aliases": len(alias_records),
        "advertised_model_types": len(all_records),
        "historical_green_profiles_discovered": historical.get("forward", {}).get("profiles"),
        "historical_green_profiles_reproduced": (
            len({row["family"] for row in records})
            if all(row.get("pass") is True for row in records)
            else 0
        ),
        "historical_native_architectures_preserved": sum(
            1
            for row in matrix_records
            if row["overall_status"] == "historical_pass_preserved"
        ),
        "historical_regressions": sorted(
            row["model_type"] for row in matrix_records if row["overall_status"] == "regression"
        ),
        "functional_smoke_coverage": sum(
            1 for row in matrix_records if row["overall_status"] == "functional_smoke_only"
        ),
        "numerical_parity_coverage": sum(
            1
            for row in matrix_records
            if row["overall_status"]
            in {"historical_pass_preserved", "newly_qualified"}
        ),
        "training_profiles": len(peft_profile_names),
        "generation_requests": generation_rows,
        "unsupported_or_unmeasured": sorted(
            row["model_type"]
            for row in matrix_records
            if row["overall_status"] in {"unsupported_capability", "not_tested", "blocked"}
        ),
        "status_counts": {
            status: sum(1 for row in all_records if row["overall_status"] == status)
            for status in sorted(STATUSES)
        },
    }
    evidence = []
    for path in [historical_path, forward_path, peft_path, generation_path]:
        if path and Path(path).is_file():
            evidence.append(
                {
                    "path": str(Path(path).relative_to(ROOT)),
                    "sha256": sha(Path(path)),
                }
            )
    matrix = {
        "schema": 1,
        "phase": "XIX-W",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": _source_revision(),
        "qualified_instrumented_source": "60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab",
        "stable_revision": "0483bcab17c0ad01872afb8bd5aa8bdc90e6b657",
        "registry_source_sha256": registry["source_sha256"],
        "policy": POLICY,
        "device": {
            "name": "AMD Radeon Graphics / Ryzen Z1 Extreme",
            "vendor_id": 4098,
            "device_id": 5567,
            "isa": "RDNA3",
        },
        "evidence": evidence,
        "historical_native_architectures": sorted(historical_native),
        "historical_forward_profiles": sorted({row["family"] for row in records}),
        "alias_targets": {
            row["model_type"]: row["canonical_model_type"] for row in alias_records
        },
        "summary": summary,
        "records": all_records,
        "forward_records": [
            {
                "family": row["family"],
                "architecture": row["architecture"],
                "case": row["case"],
                "mode": row["mode"],
                "pass": row["pass"],
                "max_abs": row.get("max_abs"),
                "failing_values": row.get("failing_values"),
                "device_id": row.get("device_id"),
            }
            for row in records
        ],
        "generation_records": [
            {
                "architecture": architecture,
                "runs": len(rows),
                "pass": all(row.get("pass") is True for row in rows),
            }
            for architecture, rows in sorted(generation.items())
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return matrix


def validate(matrix: dict, registry: dict | None = None) -> None:
    registry = registry or parse_registry()
    if matrix.get("phase") != "XIX-W":
        raise ValueError("not a Phase XIX-W matrix")
    if matrix.get("policy") != POLICY:
        raise ValueError("matrix numerical policy changed")
    rows = matrix.get("records", [])
    if len(rows) != TOTAL_COUNT:
        raise ValueError("matrix row count is not 229")
    ids = [row.get("model_type") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate matrix rows")
    if {row["model_type"] for row in rows if row.get("is_alias")} != set(registry["aliases"]):
        raise ValueError("alias inventory is incomplete")
    if {row["model_type"] for row in rows if not row.get("is_alias")} != set(registry["canonical"]):
        raise ValueError("canonical inventory is incomplete")
    if any(row.get("overall_status") not in STATUSES for row in rows):
        raise ValueError("unknown compatibility status")
    historical = set(matrix.get("historical_native_architectures", []))
    if len(historical) != 26:
        raise ValueError("historical architecture set is not the sealed 26")
    alias_targets = matrix.get("alias_targets", {})
    if set(alias_targets) != set(registry["aliases"]):
        raise ValueError("alias target map is incomplete")
    for alias in registry["aliases"]:
        expected = _alias_target(registry["alias_targets"].get(alias))
        if alias_targets.get(alias) != expected:
            raise ValueError(f"alias resolves to the wrong graph: {alias}")
    canonical = set(registry["canonical"])
    for target in alias_targets.values():
        if target != "nested_text_config" and target not in canonical:
            raise ValueError(f"alias target is not canonical: {target}")
    for architecture in historical:
        row = next(row for row in rows if row["model_type"] == architecture and not row["is_alias"])
        if row["overall_status"] != "historical_pass_preserved":
            raise ValueError(f"historical architecture is not preserved: {architecture}")
        if not row.get("forward_result", {}).get("pass"):
            raise ValueError(f"historical architecture lacks a passing forward result: {architecture}")
    if any(row["overall_status"] in {"regression", "blocked"} for row in rows):
        raise ValueError("matrix contains a regression or blocked row")
    for row in rows:
        if row["overall_status"] == "not_tested" and row.get("forward_result"):
            raise ValueError(f"untested architecture claims a forward result: {row['model_type']}")
        if row["is_alias"] and row.get("forward_result"):
            raise ValueError(f"alias claims an independent numerical pass: {row['model_type']}")
        if row["overall_status"] == "historical_pass_preserved" and row["model_type"] not in historical:
            raise ValueError(f"non-historical architecture claims preserved parity: {row['model_type']}")
    forward = matrix.get("forward_records", [])
    if len(forward) != 166 or not all(row.get("pass") is True for row in forward):
        raise ValueError("current forward replay is not the complete 166-row green matrix")
    cases = {(row.get("family"), row.get("case")) for row in forward}
    if len(cases) != 83 or {row.get("mode") for row in forward} != {"resident", "streaming"}:
        raise ValueError("forward replay does not cover 83 cases in both storage modes")
    if any(
        row.get("max_abs", 1.0) >= POLICY["absolute_limit"]
        or row.get("failing_values") != 0
        for row in forward
    ):
        raise ValueError("forward replay contains a strict-threshold violation")
    if any(row.get("device_id") != 0x15BF for row in forward):
        raise ValueError("forward replay is not bound to physical RDNA3")
    if matrix.get("summary", {}).get("generation_requests") != 72:
        raise ValueError("generation matrix is not 72 requests")
    for evidence in matrix.get("evidence", []):
        path = ROOT / evidence["path"]
        if not path.is_file() or sha(path) != evidence["sha256"]:
            raise ValueError(f"missing or altered evidence: {evidence['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--peft", type=Path, default=DEFAULT_PEFT)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate the committed matrix without rebuilding it")
    args = parser.parse_args()
    if args.check:
        matrix = read_json(args.output)
        validate(matrix)
        print(
            f"validated {args.output}: {matrix['summary']['canonical_types']} canonical + "
            f"{matrix['summary']['aliases']} aliases"
        )
        return 0
    matrix = build(
        args.forward, args.historical, args.peft, args.generation, args.output
    )
    if args.validate:
        validate(matrix)
    print(
        f"wrote {args.output}: {matrix['summary']['canonical_types']} canonical + "
        f"{matrix['summary']['aliases']} aliases, "
        f"{matrix['summary']['historical_green_profiles_reproduced']} historical preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and validate the Phase XIX-W per-architecture compatibility matrix.

The matrix is source-bound to the Rust registry, the production source, the
sealed receipts, and the current replay evidence. It never infers numerical
qualification from registry membership: a canonical architecture is
``historical_pass_preserved`` only when every current replay row for it passes.
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
ORACLE_TRANSFORMERS_REVISION = "129e8dc306932532e0b9e702d1fb4d5d7771c2a4"
PEFT_BINARIES = frozenset({
    "hierarchos-vulkan-transformer-logits.exe",
    "transformer_parity.exe",
    "peft_checkpoint.exe",
})

# Sealed historical coverage. These names are the regression contract, not a
# coverage percentage: a missing name is a regression even if another profile
# happens to produce the same total count.
HISTORICAL_FORWARD_PROFILES = frozenset({
    "bart", "bert", "dbrx", "deepseek_v4",
    "falcon_h1_gated_after", "falcon_h1_gated_before", "falcon_h1_silu",
    "gemma3", "gemma3_softcap_tied", "gemma4", "gpt2", "gpt2_wide",
    "gpt_oss", "kimi_k25_text", "kimi_k3_kda", "kimi_k3_mla", "kimi_k3_text",
    "llama", "llama_wide", "minimax_m2", "minimax_m3", "minimax_m3_dense",
    "mistral", "mistral4", "mixtral", "phi3", "phi4_multimodal_text",
    "qwen2", "qwen2_5_gqa", "qwen2_5_sliding_tied", "qwen3_5_full",
    "qwen3_5_linear", "qwen3_5_mixed", "qwen3_5_moe", "qwen3_next",
    "qwen4_exp_linear", "qwen4_exp_mixed_ple", "qwen4_exp_ple",
    "qwen4_exp_qsa", "smollm3", "smollm3_yarn", "switch_transformers", "t5",
})
HISTORICAL_NATIVE_ARCHITECTURES = frozenset({
    "bart", "bert", "dbrx", "deepseek_v3", "deepseek_v4", "falcon_h1",
    "gemma3_text", "gemma4_text", "gpt2", "gpt_oss", "kimi_linear", "llama",
    "minimax_m2", "minimax_m3_vl_text", "mistral", "mistral4", "mixtral",
    "phi3", "qwen2", "qwen3_5_moe_text", "qwen3_5_text", "qwen3_next",
    "qwen4_exp_text", "smollm3", "switch_transformers", "t5",
})
HISTORICAL_GENERATION_PROFILES = frozenset({
    "gpt2", "llama", "mistral", "mixtral", "qwen2", "qwen3_next", "gpt_oss",
    "smollm3", "qwen2_5_gqa", "qwen2_5_sliding_tied", "qwen3_5_full",
    "qwen3_5_linear", "qwen3_5_mixed", "qwen3_5_moe", "qwen4_exp_linear",
    "qwen4_exp_qsa", "qwen4_exp_ple", "qwen4_exp_mixed_ple",
})
HISTORICAL_TRAINING_PROFILES = frozenset({
    "gpt2", "llama", "mixtral", "qwen3_next", "deepseek_v4",
    "phi4_multimodal_text", "phi3", "kimi_k25_text", "kimi_k3_kda",
    "kimi_k3_mla", "kimi_k3_text", "gpt_oss", "smollm3", "qwen2_5_gqa",
    "qwen2_5_sliding_tied", "qwen3_5_full", "qwen3_5_linear",
    "qwen3_5_mixed", "qwen3_5_moe", "qwen4_exp_linear", "qwen4_exp_qsa",
    "qwen4_exp_ple", "qwen4_exp_mixed_ple", "mistral4", "minimax_m3_dense",
    "minimax_m3", "gemma4", "minimax_m2", "gemma3",
    "falcon_h1_gated_after", "falcon_h1_gated_before", "falcon_h1_silu",
})

# A PEFT profile is a text graph surface. The canonical target is the graph
# that the independent harness constructs; it is not a claim that every other
# alias or multimodal component has the same training capability.
PEFT_PROFILE_TARGETS = {
    "gpt2": "gpt2",
    "llama": "llama",
    "mixtral": "mixtral",
    "qwen3_next": "qwen3_next",
    "deepseek_v4": "deepseek_v4",
    "phi4_multimodal_text": "phi3",
    "phi3": "phi3",
    "kimi_k25_text": "deepseek_v3",
    "kimi_k3_kda": "kimi_linear",
    "kimi_k3_mla": "kimi_linear",
    "kimi_k3_text": "kimi_linear",
    "gpt_oss": "gpt_oss",
    "smollm3": "smollm3",
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
    "mistral4": "mistral4",
    "minimax_m3_dense": "minimax_m3_vl_text",
    "minimax_m3": "minimax_m3_vl_text",
    "gemma4": "gemma4_text",
    "minimax_m2": "minimax_m2",
    "gemma3": "gemma3_text",
    "falcon_h1_gated_after": "falcon_h1",
    "falcon_h1_gated_before": "falcon_h1",
    "falcon_h1_silu": "falcon_h1",
}

SOURCE_FILES = {
    "transformer_sha256": ROOT / "hierarchos-vulkan/src/transformer.rs",
    "reciprocal_shader_sha256": ROOT
    / "hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv",
    "registry_parser_sha256": ROOT / "hierarchos-vulkan/validation/phase19w_registry.py",
    "matrix_builder_sha256": ROOT / "hierarchos-vulkan/validation/phase19w_matrix.py",
    "forward_producer_sha256": ROOT
    / "hierarchos-vulkan/validation/qualify_gen9_architectures.py",
    "generation_producer_sha256": ROOT
    / "hierarchos-vulkan/validation/qualify_gen9_generation.py",
    "peft_producer_sha256": ROOT
    / "hierarchos-vulkan/validation/qualify_parameter_streaming.py",
    "peft_oracle_sha256": ROOT
    / "hierarchos-vulkan/validation/verify_peft_lora_strict.py",
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


def source_contract() -> dict:
    return {
        "revision": _source_revision(),
        **{name: sha(path) for name, path in SOURCE_FILES.items()},
    }


def _validate_source_contract(contract: dict) -> None:
    expected = source_contract()
    for name, value in expected.items():
        if name == "revision":
            # A committed matrix cannot contain the hash of the commit that
            # first adds it (the commit hash is computed after its tree). The
            # file hashes below bind the exact qualified source; the revision
            # remains provenance only.
            continue
        if contract.get(name) != value:
            raise ValueError(f"matrix source contract changed: {name}")


def _forward_rows(path: Path) -> list[dict]:
    value = read_json(path)
    records = value.get("records", [])
    if not records:
        raise ValueError(f"no forward records in {path}")
    return records


def _validate_forward(forward: dict, records: list[dict], historical: dict) -> None:
    if (forward.get("absolute_limit"), forward.get("rtol"), forward.get("comparison")) != (
        POLICY["absolute_limit"], POLICY["rtol"], POLICY["comparison"]
    ):
        raise ValueError("current forward policy differs from Phase XIX-W")
    if len(records) != 166:
        raise ValueError(f"forward replay has {len(records)} rows, expected 166")
    if not forward.get("pass") or any(row.get("pass") is not True for row in records):
        raise ValueError("current forward replay contains a failing row")
    profiles = {row.get("family") for row in records}
    if profiles != HISTORICAL_FORWARD_PROFILES:
        raise ValueError(
            "forward profile set differs from the sealed 43: "
            f"missing={sorted(HISTORICAL_FORWARD_PROFILES - profiles)} "
            f"extra={sorted(profiles - HISTORICAL_FORWARD_PROFILES)}"
        )
    cases = {(row.get("family"), row.get("case")) for row in records}
    if len(cases) != 83:
        raise ValueError(f"forward replay has {len(cases)} cases, expected 83")
    pairs: dict[tuple, set[str]] = {}
    for row in records:
        key = (row.get("family"), row.get("case"))
        mode = row.get("mode")
        if mode not in {"resident", "streaming"}:
            raise ValueError("forward record has an unknown execution mode")
        pairs.setdefault(key, set()).add(mode)
        if row.get("vendor_id") != 0x1002 or row.get("device_id") != 0x15BF:
            raise ValueError("forward replay is not from physical RDNA3 1002:15bf")
        if row.get("max_abs", 1.0) >= POLICY["absolute_limit"]:
            raise ValueError("forward replay violates the strict numerical limit")
        if row.get("failing_values") != 0:
            raise ValueError("forward replay reports failing values")
        if mode == "streaming" and row.get("resident_word_differences") != 0:
            raise ValueError("forward resident/streaming words differ")
    if any(modes != {"resident", "streaming"} for modes in pairs.values()):
        raise ValueError("forward case is missing an execution mode")
    architectures = {row.get("architecture") for row in records}
    if architectures != HISTORICAL_NATIVE_ARCHITECTURES:
        raise ValueError(
            "forward native architecture set differs from the sealed 26: "
            f"missing={sorted(HISTORICAL_NATIVE_ARCHITECTURES - architectures)} "
            f"extra={sorted(architectures - HISTORICAL_NATIVE_ARCHITECTURES)}"
        )
    historical_forward = historical.get("forward", {})
    if (
        historical_forward.get("profiles"),
        historical_forward.get("cases"),
        historical_forward.get("runs"),
    ) != (43, 83, 166):
        raise ValueError("historical receipt does not describe 43/83/166 forward coverage")
    if set(historical_forward.get("native_architectures", [])) != HISTORICAL_NATIVE_ARCHITECTURES:
        raise ValueError("historical native architecture set is not the sealed 26")


def _validate_generation(path: Path | None, family_to_architecture: dict[str, str]) -> tuple[dict, dict[str, list[dict]]]:
    if path is None or not Path(path).is_file():
        raise ValueError("generation evidence is required for the Phase XIX-W matrix")
    value = read_json(Path(path))
    records = value.get("records", [])
    if (value.get("absolute_limit"), value.get("rtol"), value.get("comparison")) != (
        POLICY["absolute_limit"], POLICY["rtol"], POLICY["comparison"]
    ):
        raise ValueError("generation policy differs from Phase XIX-W")
    if len(records) != 72:
        raise ValueError(f"generation replay has {len(records)} rows, expected 72")
    if not value.get("pass") or any(row.get("pass") is not True for row in records):
        raise ValueError("generation replay contains a failing request")
    profiles = {row.get("family") for row in records}
    if profiles != HISTORICAL_GENERATION_PROFILES:
        raise ValueError(
            "generation profile set differs from the sealed 18: "
            f"missing={sorted(HISTORICAL_GENERATION_PROFILES - profiles)} "
            f"extra={sorted(profiles - HISTORICAL_GENERATION_PROFILES)}"
        )
    grouped: dict[str, list[dict]] = {}
    combos: dict[str, set[tuple[bool, str]]] = {}
    for row in records:
        family = row.get("family")
        if family not in family_to_architecture:
            raise ValueError(f"generation family has no forward architecture: {family}")
        mode = row.get("mode")
        cached = row.get("use_cache")
        if mode not in {"resident", "streaming"} or type(cached) is not bool:
            raise ValueError("generation row has an invalid cache or storage mode")
        combos.setdefault(family, set()).add((cached, mode))
        if row.get("max_abs", 1.0) >= POLICY["absolute_limit"]:
            raise ValueError("generation replay violates the strict numerical limit")
        if mode == "streaming" and row.get("resident_word_differences") != 0:
            raise ValueError("generation resident/streaming words differ")
        if mode == "streaming" and not isinstance(row.get("streamed_linears"), int):
            raise ValueError("generation streaming row lacks streamed-linear evidence")
        if mode == "streaming" and row["streamed_linears"] <= 0:
            raise ValueError("generation streaming row executed no streamed linears")
        if not isinstance(row.get("sequence"), list) or not row["sequence"]:
            raise ValueError("generation row lacks a generated sequence")
        grouped.setdefault(family_to_architecture[family], []).append(row)
    for family, values in combos.items():
        if values != {(False, "resident"), (False, "streaming"),
                      (True, "resident"), (True, "streaming")}:
            raise ValueError(f"generation family lacks all four requests: {family}")
    return value, grouped


def _peft_report_max_abs(report: dict) -> float:
    """Read the ceiling from every strict metric in the oracle report."""
    values = []
    for source in (report, report.get("metrics") or {}, report.get("lifecycle") or {}):
        for key, value in source.items():
            if key.endswith("_max_abs") and isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        raise ValueError("PEFT report contains no strict numerical metric")
    return max(values)


def _peft_rows(path: Path) -> tuple[dict, list[dict]]:
    value = read_json(path)
    if "results" in value:
        profiles = {}
        for profile, row in value["results"].items():
            report = row.get("report") or {}
            if report:
                if report.get("family") != profile:
                    raise ValueError(f"PEFT oracle report family mismatch: {profile}")
                if report.get("peft_tolerance") != POLICY["absolute_limit"] or report.get("rtol") != POLICY["rtol"]:
                    raise ValueError(f"PEFT oracle policy changed: {profile}")
                if report.get("comparison") != POLICY["comparison"]:
                    raise ValueError(f"PEFT oracle comparison changed: {profile}")
                if report.get("transformers_revision") != ORACLE_TRANSFORMERS_REVISION:
                    raise ValueError(f"PEFT oracle checkout revision changed: {profile}")
                if report.get("parameter_streaming") is not True:
                    raise ValueError(f"PEFT profile did not request parameter streaming: {profile}")
            profiles[profile] = {
                "profile": profile,
                "pass": bool(row.get("exit_code") == 0 and report.get("pass") is True),
                "max_abs": _peft_report_max_abs(report) if report else None,
                "streamed_linears": row.get("streamed_linears"),
                "source_first": bool(row.get("source_first_parameters")),
                "oracle_provenance": report.get("transformers_source") if report else None,
                "transformers_revision": report.get("transformers_revision") if report else None,
                "peft_tolerance": report.get("peft_tolerance") if report else None,
            }
    else:
        profiles = {}
        for profile, row in value.get("profiles", {}).items():
            profiles[profile] = {
                "profile": profile,
                "pass": bool(value.get("status") == "qualified" and row.get("pass") is not False),
                "max_abs": row.get("max_abs"),
                "streamed_linears": row.get("streamed_linears"),
                "source_first": bool(row.get("source_first")),
            }
    names = set(profiles)
    if names != HISTORICAL_TRAINING_PROFILES:
        raise ValueError(
            "PEFT profile set differs from the sealed 32: "
            f"missing={sorted(HISTORICAL_TRAINING_PROFILES - names)} "
            f"extra={sorted(names - HISTORICAL_TRAINING_PROFILES)}"
        )
    for profile, row in profiles.items():
        if not row["pass"]:
            raise ValueError(f"PEFT profile failed: {profile}")
        if not isinstance(row["max_abs"], (int, float)) or row["max_abs"] >= POLICY["absolute_limit"]:
            raise ValueError(f"PEFT profile violates the strict numerical limit: {profile}")
        if profile not in PEFT_PROFILE_TARGETS:
            raise ValueError(f"PEFT profile has no canonical target: {profile}")
        if not isinstance(row["streamed_linears"], int) or row["streamed_linears"] <= 0:
            raise ValueError(f"PEFT profile lacks streamed-linear evidence: {profile}")
    binaries = value.get("binaries")
    if not isinstance(binaries, dict) or set(binaries) != PEFT_BINARIES:
        raise ValueError("PEFT evidence does not bind the complete three-binary set")
    for name, digest_value in binaries.items():
        if not isinstance(digest_value, str) or len(digest_value) != 64:
            raise ValueError(f"invalid PEFT executable hash: {name}")
    return value, [profiles[name] for name in sorted(profiles)]


def _validate_executable_evidence(forward: dict, generation: dict, peft: dict) -> dict:
    evidence = {
        "forward": forward.get("binary_sha256"),
        "generation": generation.get("binary_sha256"),
        "peft": peft.get("binaries"),
    }
    if not evidence["forward"] or not evidence["generation"]:
        raise ValueError("forward or generation evidence lacks an executable hash")
    return evidence


def _peft_by_architecture(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(PEFT_PROFILE_TARGETS[row["profile"]], []).append(row)
    return grouped


def _generation_by_architecture(
    value: dict | None, family_to_architecture: dict[str, str]
) -> dict[str, list[dict]]:
    if not value:
        return {}
    grouped: dict[str, list[dict]] = {}
    for row in value.get("records", []):
        architecture = family_to_architecture.get(row.get("family"))
        if not architecture:
            raise ValueError("generation record has no architecture mapping")
        grouped.setdefault(architecture, []).append(row)
    return grouped


def _alias_target(target: str | None) -> str:
    return target if target else "nested_text_config"


def _evidence_entry(path: Path | None) -> list[dict]:
    result = []
    if path is None:
        return result
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing evidence file: {path}")
    result.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
    return result


def build(
    forward_path: Path,
    historical_path: Path,
    peft_path: Path,
    generation_path: Path,
    output: Path,
) -> dict:
    registry = parse_registry()
    historical = read_json(historical_path)
    forward = read_json(forward_path)
    records = _forward_rows(forward_path)
    _validate_forward(forward, records, historical)
    by_architecture: dict[str, list[dict]] = {}
    for row in records:
        by_architecture.setdefault(row["architecture"], []).append(row)
    family_to_architecture = {row["family"]: row["architecture"] for row in records}
    generation_value, _generation = _validate_generation(generation_path, family_to_architecture)
    generation = _generation_by_architecture(generation_value, family_to_architecture)
    peft_value, peft_rows = _peft_rows(peft_path)
    executable_evidence = _validate_executable_evidence(forward, generation_value, peft_value)
    peft = _peft_by_architecture(peft_rows)

    historical_native = set(historical["forward"]["native_architectures"])
    if historical_native != HISTORICAL_NATIVE_ARCHITECTURES:
        raise ValueError("historical native architecture set changed")
    canonical_arch = set(registry["canonical"])
    aliases = registry["aliases"]

    matrix_records = []
    for architecture in registry["canonical"]:
        rows = by_architecture.get(architecture, [])
        cases = sorted({row["family"] for row in rows})
        modes = sorted({row["mode"] for row in rows})
        passed = bool(rows) and all(row.get("pass") is True for row in rows)
        if architecture in historical_native:
            status = "historical_pass_preserved" if passed else ("blocked" if not rows else "regression")
        elif rows:
            status = "newly_qualified" if passed else "regression"
        else:
            status = "not_tested"
        training_rows = peft.get(architecture, [])
        generation_rows = generation.get(architecture, [])
        capabilities = []
        if rows:
            capabilities.extend(["config_parse", "safetensors_load", "native_vulkan_forward",
                                 "native_package_export"])
        if generation_rows:
            capabilities.append("generation")
        if training_rows:
            capabilities.extend(["lora_training", "backward"])
        historical_capabilities = []
        if architecture in historical_native:
            historical_capabilities.append("forward")
        if generation_rows:
            historical_capabilities.append("generation")
        if training_rows:
            historical_capabilities.append("training")
        training_result = None
        if training_rows:
            training_result = {
                "pass": all(row["pass"] for row in training_rows),
                "max_abs": max(row["max_abs"] for row in training_rows),
                "profiles": [row["profile"] for row in training_rows],
                "streamed_linears": {row["profile"]: row["streamed_linears"] for row in training_rows},
            }
        generation_result = None
        if generation_rows:
            generation_result = {
                "runs": len(generation_rows),
                "pass": all(row["pass"] for row in generation_rows),
                "profiles": sorted({row["family"] for row in generation_rows}),
            }
        matrix_records.append(
            {
                "model_type": architecture,
                "canonical_model_type": architecture,
                "is_alias": False,
                "architecture_family": architecture,
                "declared_capabilities": capabilities,
                "historically_qualified_capabilities": historical_capabilities,
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
                "backward_result": training_result,
                "training_result": training_result,
                "generation_result": generation_result,
                "export_reload_result": None,
                "fallback_behavior": (
                    "smollm3_full_prefix_when_cache_requested"
                    if architecture == "smollm3" else "none_declared"
                ),
                "numerical_policy": POLICY,
                "execution_evidence": (
                    f"{historical_path.name} + {forward_path.name}" if rows else "none"
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
                "declared_capabilities": (
                    ["config_parse", "text_backbone_resolution", "nested_text_config"]
                    if target == "nested_text_config"
                    else ["config_parse", "text_backbone_resolution", "canonical_text_backbone"]
                ),
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
                "fallback_behavior": "nested_text_config_selected_at_load",
                "numerical_policy": POLICY,
                "execution_evidence": "source_alias_route_verified; no numerical alias oracle",
                "alias_constructors": registry["alias_constructors"].get(alias, []),
                "alias_resolution_result": target,
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
    if len(matrix_records) != CANONICAL_COUNT or len(alias_records) != ALIAS_COUNT:
        raise ValueError("registry counts do not match 146 canonical plus 83 aliases")

    summary = {
        "canonical_types": len(matrix_records),
        "aliases": len(alias_records),
        "advertised_model_types": len(all_records),
        "historical_forward_profiles": len(HISTORICAL_FORWARD_PROFILES),
        "historical_forward_profiles_reproduced": len({row["family"] for row in records}),
        "historical_forward_cases": len({(row["family"], row["case"]) for row in records}),
        "historical_forward_runs": len(records),
        "historical_native_architectures": len(historical_native),
        "historical_green_profiles_discovered": historical["forward"]["profiles"],
        "historical_green_profiles_reproduced": len({row["family"] for row in records}),
        "historical_native_architectures_preserved": sum(
            1 for row in matrix_records if row["overall_status"] == "historical_pass_preserved"
        ),
        "historical_training_profiles": len(HISTORICAL_TRAINING_PROFILES),
        "historical_training_profiles_reproduced": len(peft_rows),
        "historical_regressions": sorted(
            row["model_type"] for row in matrix_records if row["overall_status"] == "regression"
        ),
        "functional_smoke_coverage": sum(
            1 for row in matrix_records if row["overall_status"] == "functional_smoke_only"
        ),
        "numerical_parity_coverage": sum(
            1
            for row in matrix_records
            if row["overall_status"] in {"historical_pass_preserved", "newly_qualified"}
        ),
        "training_profiles": len(peft_rows),
        "generation_profiles": len(HISTORICAL_GENERATION_PROFILES),
        "generation_requests": len(generation_value["records"]),
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
        evidence.extend(_evidence_entry(path))
    matrix = {
        "schema": 1,
        "phase": "XIX-W",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": _source_revision(),
        "qualified_instrumented_source": _source_revision(),
        "historical_qualified_source": "60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab",
        "stable_revision": "0483bcab17c0ad01872afb8bd5aa8bdc90e6b657",
        "source_contract": source_contract(),
        "registry_source_sha256": registry["source_sha256"],
        "policy": POLICY,
        "device": {
            "name": "AMD Radeon Graphics / Ryzen Z1 Extreme",
            "vendor_id": 4098,
            "device_id": 5567,
            "isa": "RDNA3",
        },
        "evidence": evidence,
        "executable_evidence": executable_evidence,
        "historical_forward_profiles": sorted(HISTORICAL_FORWARD_PROFILES),
        "historical_native_architectures": sorted(historical_native),
        "historical_training_profiles": sorted(HISTORICAL_TRAINING_PROFILES),
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
                "word_differences": row.get("word_differences"),
                "resident_word_differences": row.get("resident_word_differences"),
                "device_id": row.get("device_id"),
                "vendor_id": row.get("vendor_id"),
                "streamed_linears": row.get("streamed_linears"),
            }
            for row in records
        ],
        "generation_records": [
            {
                "family": row["family"],
                "use_cache": row["use_cache"],
                "mode": row["mode"],
                "pass": row["pass"],
                "max_abs": row.get("max_abs"),
                "sequence": row.get("sequence"),
                "resident_word_differences": row.get("resident_word_differences"),
            }
            for row in generation_value["records"]
        ],
        "training_records": peft_rows,
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
    _validate_source_contract(matrix.get("source_contract", {}))
    if matrix.get("registry_source_sha256") != registry["source_sha256"]:
        raise ValueError("matrix registry source changed")
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
    if historical != HISTORICAL_NATIVE_ARCHITECTURES:
        raise ValueError("historical architecture set is not the sealed 26")
    if set(matrix.get("historical_forward_profiles", [])) != HISTORICAL_FORWARD_PROFILES:
        raise ValueError("historical forward profile set is not the sealed 43")
    if set(matrix.get("historical_training_profiles", [])) != HISTORICAL_TRAINING_PROFILES:
        raise ValueError("historical training profile set is not the sealed 32")
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
        if row["overall_status"] == "newly_qualified" and not row.get("forward_result"):
            raise ValueError(f"newly qualified architecture lacks forward evidence: {row['model_type']}")
    forward = matrix.get("forward_records", [])
    if len(forward) != 166 or not all(row.get("pass") is True for row in forward):
        raise ValueError("current forward replay is not the complete 166-row green matrix")
    cases = {(row.get("family"), row.get("case")) for row in forward}
    if len(cases) != 83 or {row.get("mode") for row in forward} != {"resident", "streaming"}:
        raise ValueError("forward replay does not cover 83 cases in both storage modes")
    if {row.get("family") for row in forward} != HISTORICAL_FORWARD_PROFILES:
        raise ValueError("forward replay profile set is incomplete")
    if {row.get("architecture") for row in forward} != HISTORICAL_NATIVE_ARCHITECTURES:
        raise ValueError("forward replay architecture set is incomplete")
    if any(
        row.get("max_abs") is None
        or row["max_abs"] >= POLICY["absolute_limit"]
        or row.get("failing_values") != 0
        or row.get("device_id") != 0x15BF
        or row.get("vendor_id") != 0x1002
        for row in forward
    ):
        raise ValueError("forward replay contains a strict-threshold or device violation")
    if any(
        row.get("mode") == "streaming" and row.get("resident_word_differences") != 0
        for row in forward
    ):
        raise ValueError("forward replay lacks resident/streaming word identity")
    generation = matrix.get("generation_records", [])
    if len(generation) != 72:
        raise ValueError("generation matrix is not 72 requests")
    generation_profiles = {row.get("family") for row in generation}
    if generation_profiles != HISTORICAL_GENERATION_PROFILES:
        raise ValueError("generation profile inventory is incomplete")
    combos: dict[str, set[tuple[bool, str]]] = {}
    for row in generation:
        if row.get("pass") is not True or row.get("max_abs") is None:
            raise ValueError("generation matrix contains a failing request")
        if row["max_abs"] >= POLICY["absolute_limit"]:
            raise ValueError("generation matrix violates the strict numerical limit")
        if row.get("mode") not in {"resident", "streaming"} or type(row.get("use_cache")) is not bool:
            raise ValueError("generation matrix has an invalid mode")
        if row.get("mode") == "streaming" and row.get("resident_word_differences") != 0:
            raise ValueError("generation matrix lacks storage word identity")
        combos.setdefault(row["family"], set()).add((row["use_cache"], row["mode"]))
        if not row.get("sequence"):
            raise ValueError("generation matrix lacks a generated sequence")
    if any(values != {(False, "resident"), (False, "streaming"),
                      (True, "resident"), (True, "streaming")}
           for values in combos.values()):
        raise ValueError("generation family lacks cache/full-prefix and storage coverage")
    training = matrix.get("training_records", [])
    if len(training) != 32:
        raise ValueError("training matrix is not the complete 32 profiles")
    if set(matrix.get("executable_evidence", {}).get("peft", {})) != PEFT_BINARIES:
        raise ValueError("matrix does not bind the complete PEFT executable set")
    if {row.get("profile") for row in training} != HISTORICAL_TRAINING_PROFILES:
        raise ValueError("training profile inventory is incomplete")
    if any(
        row.get("pass") is not True
        or row.get("max_abs") is None
        or row["max_abs"] >= POLICY["absolute_limit"]
        or not isinstance(row.get("streamed_linears"), int)
        or row["streamed_linears"] <= 0
        for row in training
    ):
        raise ValueError("training matrix contains a failed or unmeasured profile")
    summary = matrix.get("summary", {})
    if summary.get("generation_requests") != 72:
        raise ValueError("generation summary is not 72 requests")
    if summary.get("training_profiles") != 32:
        raise ValueError("training summary is not 32 profiles")
    if summary.get("historical_regressions"):
        raise ValueError("summary reports historical regressions")
    for evidence in matrix.get("evidence", []):
        # Matrices built on Windows before the POSIX-path fix stored native
        # separators. Normalize those without changing which file is bound.
        path = ROOT / Path(str(evidence["path"]).replace("\\", "/"))
        if not path.is_file() or sha(path) != evidence["sha256"]:
            raise ValueError(f"missing or altered evidence: {evidence['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--peft", type=Path, default=DEFAULT_PEFT)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate the committed matrix without rebuilding it")
    args = parser.parse_args()
    if not args.check and (args.forward is None or args.generation is None):
        parser.error("--forward and --generation are required unless --check is used")
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

#!/usr/bin/env python3
"""Shared Phase XIX-T verification-policy helpers.

Phase XIX-S remains the frozen numerical receipt. This module defines the
versioned verification policy that evaluates that receipt without changing its
bound source or evidence bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT_DIR = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261110-phase19s-current-source"
)
DEFAULT_RECEIPT = RECEIPT_DIR / "current-qualification.json"
DEFAULT_OVERLAY = CRATE / "validation/phase19t-policy-overlay.json"

EVIDENCE_SCHEMA = 1
EVIDENCE_KIND = "phase19t-policy-overlay"
NUMERICAL_KIND = "phase19s-instrumented-current-source"
NUMERICAL_SOURCE_REVISION = "60063fd3b0a3a963b05e7d5c472cf9b27a4a1cab"
EXPECTED_SUBREVISIONS = {
    "attention": "1561560c8606640f69efc6580b8401411f93399a",
    "training": "8801f76461ba8c783b2c78f38b2b4f7912b3d95f",
    "stability": "271fdddfdbab06bbb20b11105d69d12b8b2cc13d",
    "heldout": "111f5427394aa5aba05720f5361943406f7b88d1",
    "drift": "d22775dc29661f323b80a5a755883b35d9875385",
    "activation": "1a1cfe2542324e0036164d56afb7b534d4b2d715",
}
EXPECTED_NUMERICAL_SOURCE_COUNT = 165
EXPECTED_SOURCE_COUNT = 178
EXPECTED_EVIDENCE_COUNT = 338
EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256 = (
    "e55a7544c452bbaf38219dce6023f34f5ba14c4370e18220f186096ee11733d8"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "ec8413d86b88ee98df914eaa415002c0c383544e42651e8ac67ce36c118c56c8"
)
EXPECTED_EVIDENCE_MANIFEST_SHA256 = (
    "16457422f6ee1555e6c27f842ca7de62da1269865cef248bd6cd332931d9808e"
)
EXPECTED_FAMILY_COUNTS = {
    "attention": 16,
    "training": 136,
    "stability": 121,
    "heldout": 27,
    "drift": 20,
    "activation": 18,
}
EXPECTED_SAFETENSOR_COUNT = 36
EXPECTED_SOURCE_GROUPS = {
    "hierarchos-vulkan/src/": 19,
    "hierarchos-vulkan/shaders/": 98,
    "hierarchos-vulkan/validation/": 59,
}
CRITICAL_SOURCES = (
    "hierarchos-vulkan/Cargo.toml",
    "hierarchos-vulkan/Cargo.lock",
    "hierarchos-vulkan/src/transformer.rs",
    "hierarchos-vulkan/src/wmma.rs",
    "hierarchos-vulkan/src/wmma_linear.rs",
    "hierarchos-vulkan/src/activation_trace.rs",
    "hierarchos-vulkan/src/bin/activation_probe.rs",
    "hierarchos-vulkan/src/bin/wmma_drift.rs",
    "hierarchos-vulkan/src/bin/transformer_train.rs",
    "hierarchos-vulkan/src/bin/hierarchos-vulkan-training-qualification.rs",
    "hierarchos-vulkan/src/bin/training-stability.rs",
    "hierarchos-vulkan/src/bin/held_out_eval.rs",
    "hierarchos-vulkan/validation/check_phase19s_current.py",
    "hierarchos-vulkan/validation/qualify_phase19s_current.py",
    "hierarchos-vulkan/validation/test_phase19s_current.py",
    "hierarchos-vulkan/validation/check_activation_phase19r.py",
    "hierarchos-vulkan/validation/qualify_activation_phase19r.py",
    "hierarchos-vulkan/validation/test_activation_phase19r.py",
    "hierarchos-vulkan/validation/check_training_phase16.py",
    "hierarchos-vulkan/validation/check_training_stability.py",
    "hierarchos-vulkan/validation/check_heldout_phase17.py",
    "hierarchos-vulkan/validation/check_wmma_drift.py",
)
REQUIRED_BINARIES = (
    "image_text_pipeline_probe",
    "gemma3_vision_probe",
    "gemma4_vision_probe",
    "llava_pipeline_probe",
    "activation_probe",
    "wmma_drift",
    "training_harness",
    "trainer",
    "stability_harness",
    "held_out_eval",
    "libtest",
)
BINARY_FILES = {
    "image_text_pipeline_probe": "image_text_pipeline_probe.exe",
    "gemma3_vision_probe": "gemma3_vision_probe.exe",
    "gemma4_vision_probe": "gemma4_vision_probe.exe",
    "llava_pipeline_probe": "llava_pipeline_probe.exe",
    "activation_probe": "activation_probe.exe",
    "wmma_drift": "wmma_drift.exe",
    "training_harness": "hierarchos-vulkan-training-qualification.exe",
    "trainer": "hierarchos-vulkan-transformer-train.exe",
    "stability_harness": "training-stability.exe",
    "held_out_eval": "held_out_eval.exe",
    "libtest": "hierarchos_vulkan-8c8d2140efff5506.exe",
}
# The test executable is not built by CI; it is bound, but its bytes are only
# checked when a caller supplies a binary root that actually contains it.
POLICY_FILES = (
    "hierarchos-vulkan/validation/phase19t_policy.py",
    "hierarchos-vulkan/validation/check_phase19t_current.py",
    "hierarchos-vulkan/validation/qualify_phase19t_policy.py",
    "hierarchos-vulkan/validation/test_phase19t_current.py",
    "hierarchos-vulkan/validation/PHASE19S_CURRENT_SOURCE.md",
    "hierarchos-vulkan/validation/PHASE19T_SEALED_TEST_AUDIT.md",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
COMMITTED_ACTIVATION_EVIDENCE = {
    "trace-initial-off",
    "trace-initial-mixed",
    "trace-step1-fp32-off",
    "trace-step1-fp32-mixed",
    "trace-step1-mixed-off",
    "trace-step1-mixed-mixed",
    "trace-step10-off",
    "trace-step10-mixed",
    "trace-step100-off",
    "trace-step100-mixed",
    "eval-initial-off",
    "eval-initial-mixed",
    "eval-step1-fp32-off",
    "eval-step1-fp32-mixed",
    "eval-step1-mixed-off",
    "eval-step1-mixed-mixed",
    "anomaly",
}
EXTERNAL_ACTIVATION_WEIGHTS = {
    "weights-initial",
    "weights-step1-fp32",
    "weights-step1-mixed",
    "weights-step10",
    "weights-step100",
}


class PolicyError(ValueError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, "git command failed: " + " ".join(args))
    return result.stdout.strip()


def manifest_sha256(files: dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(files):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(files[name]).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def safe_relative(value: object, root: Path, label: str) -> Path:
    """Return a checked relative path rooted at ``root``.

    Receipts are authored on Windows and Linux, so both separators must be
    considered hostile until normalized. Only canonical POSIX relative names
    are accepted; the resolved target must remain beneath ``root``.
    """
    require(isinstance(value, str) and bool(value), f"missing {label} path")
    require(value == value.strip(), f"whitespace in {label} path")
    require("\x00" not in value, f"nul byte in {label} path")
    require("\\" not in value, f"backslash in {label} path: {value}")
    candidate_name = PurePosixPath(value)
    require(not candidate_name.is_absolute(), f"absolute {label} path: {value}")
    require(
        all(part not in ("", ".", "..") for part in candidate_name.parts),
        f"unsafe {label} path: {value}",
    )
    require(candidate_name.as_posix() == value, f"non-canonical {label} path: {value}")
    candidate = root.joinpath(*candidate_name.parts)
    require(not candidate.is_symlink(), f"symlink {label} path: {value}")
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = root.resolve()
    except OSError as error:
        raise PolicyError(f"unresolvable {label} path: {value}: {error}") from error
    require(
        resolved == root_resolved or root_resolved in resolved.parents,
        f"{label} path escapes permitted root: {value}",
    )
    return candidate


def validate_manifest(mapping: object, label: str) -> dict[str, str]:
    require(isinstance(mapping, dict), f"missing {label} manifest")
    require(mapping, f"empty {label} manifest")
    normalized: dict[str, str] = {}
    for name, expected in mapping.items():
        safe_relative(name, ROOT, label)
        key = PurePosixPath(name).as_posix().casefold()
        require(key not in normalized, f"duplicate {label} identity: {name}")
        require(SHA256.fullmatch(str(expected)) is not None,
                f"malformed {label} digest: {name}")
        normalized[key] = name
    return dict(mapping)


def enabled_tracked_files() -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
    )
    require(output.returncode == 0, "cannot enumerate tracked files")
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in output.stdout.split(b"\0")
        if item
    }


def check_manifest_constants(
    overlay: dict, strict: bool = True
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    numerical_sources = validate_manifest(
        overlay.get("numerical_sources"), "numerical source")
    additional_sources = validate_manifest(
        overlay.get("additional_sources"), "additional source")
    sources = validate_manifest(overlay.get("required_sources"), "source")
    evidence = validate_manifest(overlay.get("required_evidence"), "evidence")
    require(len(numerical_sources) == EXPECTED_NUMERICAL_SOURCE_COUNT,
            f"numerical source count changed: {len(numerical_sources)}")
    require(len(sources) == EXPECTED_SOURCE_COUNT,
            f"required source count changed: {len(sources)}")
    require(set(sources) == set(numerical_sources) | set(additional_sources),
            "complete source inventory is not the union of numerical and additional sources")
    require(not (set(numerical_sources) & set(additional_sources)),
            "numerical and additional source inventories overlap")
    require(len(evidence) == EXPECTED_EVIDENCE_COUNT,
            f"required evidence count changed: {len(evidence)}")
    if strict:
        require(manifest_sha256(numerical_sources) == EXPECTED_NUMERICAL_SOURCE_MANIFEST_SHA256,
                "numerical source manifest changed")
        require(manifest_sha256(sources) == EXPECTED_SOURCE_MANIFEST_SHA256,
                "required source manifest changed")
        require(manifest_sha256(evidence) == EXPECTED_EVIDENCE_MANIFEST_SHA256,
                "required evidence manifest changed")
    for name in CRITICAL_SOURCES:
        require(name in sources, "mandatory source is not bound: " + name)
    for prefix, expected in EXPECTED_SOURCE_GROUPS.items():
        count = sum(1 for name in sources if name.startswith(prefix))
        require(count == expected, f"source group {prefix} count changed: {count}")
    families = {
        family: sum(1 for name in evidence if name.startswith(family + "/"))
        for family in EXPECTED_FAMILY_COUNTS
    }
    require(families == EXPECTED_FAMILY_COUNTS, "evidence family coverage changed")
    safetensors = sorted(name for name in evidence if name.endswith(".safetensors"))
    require(len(safetensors) == EXPECTED_SAFETENSOR_COUNT,
            "required safetensors count changed")
    declared = overlay.get("required_safetensors")
    require(declared == safetensors, "required safetensors manifest changed")
    require(sum(1 for name in safetensors if name.startswith("training/evidence/")) == 18,
            "training safetensors coverage changed")
    require(sum(1 for name in safetensors if name.startswith("stability/evidence/")) == 18,
            "stability safetensors coverage changed")
    return numerical_sources, sources, evidence


def check_receipt_manifests(document: dict, numerical_sources: dict[str, str],
                             evidence: dict[str, str]) -> None:
    actual_sources = document.get("source_files")
    actual_evidence = document.get("evidence")
    require(isinstance(actual_sources, dict), "receipt source manifest missing")
    require(isinstance(actual_evidence, dict), "receipt evidence manifest missing")
    require(set(actual_sources) == set(numerical_sources),
            "receipt source set differs from the numerical inventory")
    require(set(actual_evidence) == set(evidence),
            "receipt evidence set differs from the required inventory")
    for name, expected in actual_sources.items():
        require(SHA256.fullmatch(str(expected)) is not None,
                "malformed receipt source digest: " + name)
    for name, expected in actual_evidence.items():
        require(SHA256.fullmatch(str(expected)) is not None,
                "malformed receipt evidence digest: " + name)
    for name, expected in numerical_sources.items():
        require(actual_sources[name] == expected,
                "receipt source digest differs: " + name)
    for name, expected in evidence.items():
        require(actual_evidence[name] == expected,
                "receipt evidence digest differs: " + name)


def check_sources(sources: dict[str, str], tracked: set[str]) -> None:
    for name, expected in sources.items():
        path = safe_relative(name, ROOT, "source")
        require(path.is_file(), "missing bound source: " + name)
        require(path.stat().st_size > 0, "empty bound source: " + name)
        require(name in tracked, "bound source is not tracked: " + name)
        require(digest(path) == expected, "bound source changed: " + name)


def check_evidence(receipt_dir: Path, evidence: dict[str, str],
                   receipt_repo_path: str, tracked: set[str]) -> None:
    prefix = PurePosixPath(receipt_repo_path).parent
    require(str(prefix) not in ("", "."), "receipt path has no directory")
    for name, expected in evidence.items():
        path = safe_relative(name, receipt_dir, "evidence")
        require(path.is_file(), "missing evidence: " + name)
        require(path.stat().st_size > 0 or name.endswith(".err.txt"),
                "empty mandatory evidence: " + name)
        tracked_name = (prefix / name).as_posix()
        require(tracked_name in tracked, "evidence is not tracked: " + name)
        require(digest(path) == expected, "evidence changed: " + name)


def check_binaries(document: dict, binary_root: Path | None = None) -> None:
    binaries = document.get("binaries")
    require(isinstance(binaries, dict), "missing binary manifest")
    require(set(binaries) == set(REQUIRED_BINARIES), "binary binding set changed")
    for name in REQUIRED_BINARIES:
        require(SHA256.fullmatch(str(binaries.get(name))) is not None,
                "malformed binary digest: " + name)
    if binary_root is None:
        return
    require(binary_root.is_dir(), "binary root is not a directory")
    for name, filename in BINARY_FILES.items():
        candidate = binary_root / filename
        if candidate.is_file():
            require(digest(candidate) == binaries[name], "binary hash changed: " + name)


def _ancestor(revision: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", revision, descendant],
        capture_output=True,
    )
    return result.returncode == 0


def _check_revision(value: object, label: str) -> str:
    require(isinstance(value, str) and HEX40.fullmatch(value) is not None,
            "invalid " + label + " revision")
    return value


def cross_bind(document: dict, subreceipts: dict[str, dict], policy_revision: str,
               complete_sources: dict[str, str]) -> None:
    numerical = _check_revision(document.get("source_revision"), "numerical source")
    require(numerical == NUMERICAL_SOURCE_REVISION, "numerical source revision changed")
    require(_ancestor(numerical, policy_revision),
            "numerical source is not an ancestor of the policy revision")
    require(set(subreceipts) == set(EXPECTED_SUBREVISIONS), "sub-receipt set changed")

    attention = subreceipts["attention"]
    training = subreceipts["training"]
    stability = subreceipts["stability"]
    heldout = subreceipts["heldout"]
    drift = subreceipts["drift"]
    activation = subreceipts["activation"]

    for name, expected in EXPECTED_SUBREVISIONS.items():
        actual = subreceipts[name].get("source_revision") or subreceipts[name].get("source_head")
        require(_check_revision(actual, name + " source") == expected,
                f"{name} source revision changed")
        require(_ancestor(actual, numerical),
                f"{name} source revision is outside the numerical lineage")

    devices = {
        name: subreceipts[name]["device"]
        for name in ("training", "stability", "heldout", "drift", "activation")
    }
    require(len(set(devices.values())) == 1, "sub-receipt device identities differ")
    require("AMD Radeon" in next(iter(devices.values())), "unexpected GPU device identity")
    adapter = attention.get("adapter") or {}
    require(adapter.get("vendor") == "1002" and adapter.get("device") == "15bf",
            "attention adapter identity changed")

    # One model/dataset/split lineage across the training-family receipts.
    model_config = "474f2bcade245631309021744ebf67236891ac9bda99b9418d0f4e66af3be63e"
    model_weights = "841390b8f44d25f3d35d0ad97277e9fa806ebb62a43f848784c54b5759bf6927"
    dataset = "0c5d3e907ff5158d819fbca3cd67a81f0d35578a7dfe596237879683a086af9c"
    split = "60083ed5eda85efd4fd19112ac03f3c449b8c4cecec489e9b477d8090278ecd8"
    validation_partition = "1f6852bc80bdd463f5629f6555bcbb4a1060b32c8c43c675b314ea09f7ef60d8"
    for name in ("training", "stability"):
        fixture = subreceipts[name]["fixture"]
        require(fixture.get("model_config_sha256") == model_config,
                f"{name} model config identity changed")
        require(fixture.get("model_weights_sha256") == model_weights,
                f"{name} model weight identity changed")
        require(fixture.get("dataset_sha256") == dataset,
                f"{name} dataset identity changed")
        require(fixture.get("parameters") == 16782336,
                f"{name} parameter count changed")
    require(heldout["fixture"]["files"].get("phase16_dataset.jsonl") == dataset,
            "held-out dataset identity changed")
    require(heldout["fixture"]["files"].get("split_manifest.json") == split,
            "held-out split identity changed")
    require(heldout["fixture"]["model"].get("config_sha256") == model_config
            and heldout["fixture"]["model"].get("weights_sha256") == model_weights,
            "held-out model identity changed")
    require(drift["fixture"].get("model_config_sha256") == model_config
            and drift["fixture"].get("model_weights_sha256") == model_weights,
            "drift model identity changed")
    require(drift["fixture"].get("manifest_sha256") == split,
            "drift split manifest identity changed")
    require(activation["protocol"].get("model_config_sha256") == model_config
            and activation["protocol"].get("model_weights_sha256") == model_weights,
            "activation model identity changed")
    require(activation["protocol"].get("partition_sha256") == validation_partition,
            "activation validation partition changed")

    # Shared numerical policy. Training arm summaries carry per-step results,
    # while stability carries the selected learning rate; all must agree.
    for arm in stability["arms"].values():
        summary = arm["summary"]
        require(summary.get("lr") == 0.001 and summary.get("steps") == 100,
                "stability optimizer policy changed")
        require(summary.get("beta1") == 0.9 and summary.get("beta2") == 0.999
                and summary.get("eps") == 1e-8 and summary.get("weight_decay") == 0.0,
                "stability optimizer constants changed")
    protocol = heldout["protocol"]
    require(protocol.get("learning_rate") == 0.001 and protocol.get("steps") == 100,
            "held-out optimizer policy changed")
    require(protocol.get("beta1") == 0.9 and protocol.get("beta2") == 0.999
            and protocol.get("eps") == 1e-8 and protocol.get("weight_decay") == 0.0,
            "held-out optimizer constants changed")
    require(drift["protocol"].get("lr") == 0.001 and drift["protocol"].get("steps") == 100,
            "drift optimizer policy changed")
    require(drift["protocol"].get("beta1") == 0.9 and drift["protocol"].get("beta2") == 0.999
            and drift["protocol"].get("eps") == 1e-8
            and drift["protocol"].get("weight_decay") == 0.0,
            "drift optimizer constants changed")
    require(activation["protocol"].get("lr") == 0.001
            and activation["protocol"].get("beta1") == 0.9
            and activation["protocol"].get("beta2") == 0.999
            and activation["protocol"].get("eps") == 1e-8
            and activation["protocol"].get("weight_decay") == 0.0,
            "activation optimizer constants changed")

    # Binary identities: each stage must name the executable family recorded
    # by the top-level receipt; unrelated probes remain independently bound.
    binaries = document["binaries"]
    attention_binaries = attention.get("binaries") or {}
    for name in ("image_text_pipeline_probe", "gemma3_vision_probe",
                 "gemma4_vision_probe", "llava_pipeline_probe", "libtest"):
        require(attention_binaries.get(name) == binaries[name],
                f"attention binary identity changed: {name}")
    require(training.get("binary_sha256") == binaries["training_harness"],
            "training harness identity changed")
    require(training.get("trainer_binary_sha256") == binaries["trainer"],
            "training trainer identity changed")
    require(stability.get("binaries", {}).get("harness") == binaries["stability_harness"],
            "stability harness identity changed")
    require(stability.get("binaries", {}).get("trainer") == binaries["trainer"],
            "stability trainer identity changed")
    require(heldout.get("binaries", {}).get("held_out_eval") == binaries["held_out_eval"],
            "held-out harness identity changed")
    require(heldout.get("binaries", {}).get("trainer") == binaries["trainer"],
            "held-out trainer identity changed")
    require(drift.get("binaries", {}).get("wmma_drift") == binaries["wmma_drift"],
            "drift harness identity changed")
    require(activation.get("binaries", {}).get("activation_probe") == binaries["activation_probe"],
            "activation probe identity changed")
    require(activation.get("binaries", {}).get("wmma_drift") == binaries["wmma_drift"],
            "activation drift identity changed")

    # Every sub-receipt source must also appear in the top source inventory with
    # the same bytes, and every sub-receipt evidence entry must be represented
    # under its family in the top evidence inventory.
    for name, sub in subreceipts.items():
        sub_sources = sub.get("sources") if name == "attention" else sub.get("source_files")
        for relative, expected in (sub_sources or {}).items():
            repository_name = relative if (ROOT / relative).is_file() else "hierarchos-vulkan/" + relative
            require(complete_sources.get(repository_name) == expected,
                    f"{name} source binding disagrees with the complete inventory: {relative}")
        require(f"{name}/current-qualification.json" in document["evidence"],
                f"{name} sub-receipt is absent from evidence")
        if isinstance(sub.get("evidence"), dict):
            for relative, expected in sub["evidence"].items():
                top_name = f"{name}/evidence/{relative}"
                require(document["evidence"].get(top_name) == expected,
                        f"{name} evidence binding disagrees with top receipt: {relative}")
        elif name == "attention":
            for entry in sub.get("evidence") or []:
                label = entry.get("name") if isinstance(entry, dict) else entry
                require(isinstance(label, str) and (label in document["evidence"]
                        or f"{name}/evidence/{label}.json" in document["evidence"]),
                        "attention evidence binding is absent: " + str(label))

    activation_entries = activation.get("evidence") or []
    require(isinstance(activation_entries, list) and len(activation_entries) == 22,
            "activation evidence inventory changed")
    committed = set()
    external = set()
    for entry in activation_entries:
        require(isinstance(entry, dict) and isinstance(entry.get("name"), str)
                and SHA256.fullmatch(str(entry.get("sha256"))) is not None,
                "malformed activation evidence entry")
        name = entry["name"]
        if name in EXTERNAL_ACTIVATION_WEIGHTS:
            external.add(name)
            continue
        committed.add(name)
        top_name = f"activation/evidence/{name}.json"
        require(document["evidence"].get(top_name) == entry["sha256"],
                "activation evidence hash disagrees with top receipt: " + name)
    require(committed == COMMITTED_ACTIVATION_EVIDENCE,
            "committed activation evidence set changed")
    require(external == EXTERNAL_ACTIVATION_WEIGHTS,
            "external activation weight set changed")


def check_policy_files(overlay: dict) -> None:
    files = overlay.get("policy_files")
    require(isinstance(files, dict), "missing policy file manifest")
    require(set(files) == set(POLICY_FILES), "policy file set changed")
    for name, expected in files.items():
        path = safe_relative(name, ROOT, "policy")
        require(path.is_file(), "missing policy file: " + name)
        require(SHA256.fullmatch(str(expected)) is not None,
                "malformed policy digest: " + name)
        require(digest(path) == expected, "policy file changed: " + name)

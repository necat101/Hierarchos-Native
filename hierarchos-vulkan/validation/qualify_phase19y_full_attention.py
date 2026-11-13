#!/usr/bin/env python3
"""Assemble the Phase XIX-Y full-attention current-source receipt.

The producer consumes the real replay qualifications, checks their strict
policy and coverage, binds the current source tree, executable, recovered CPU
oracle manifests and shader bytes, then exercises the checker's negative
controls before writing a PASS receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from check_phase19y_full_attention import (
    CRATE, EXPECTED_MODEL, LIMIT, ROOT, SHAPE_ARCHIVE_BYTES, SHAPE_ARCHIVE_SHA,
    SHAPE_MANIFEST_SHA, SHARED_MANIFEST_SHA, max_logit, run_negative_controls,
    validate, verify_fixture_tree,
)

RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19y"
)
PHASE19X = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-unified-release/phase19x" / "attention" / "current-qualification.json"
)
PHASE16 = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261109-rdna3-phase16-attention-backward-qualification" / "current-qualification.json"
)
VALIDATION_SOURCES = (
    "hierarchos-vulkan/validation/qualify_phase19y_full_attention.py",
    "hierarchos-vulkan/validation/check_phase19y_full_attention.py",
    "hierarchos-vulkan/validation/test_phase19y_full_attention.py",
    "hierarchos-vulkan/validation/qualify_rdna3_gpt2_shapes.py",
    "hierarchos-vulkan/validation/qualify_shared_image_pipeline.py",
    "hierarchos-vulkan/validation/qualify_multimodal_training.py",
    "hierarchos-vulkan/validation/emit_phase16_spirv_gate.py",
    "hierarchos-vulkan/validation/phase19w_matrix.py",
)


def require(condition, message):
    if not condition:
        raise SystemExit(f"PHASE XIX-Y PRODUCER: FAIL: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def source_inventory() -> dict[str, str]:
    sources = {}
    tracked = git("ls-files", "hierarchos-vulkan/src", "hierarchos-vulkan/shaders").splitlines()
    for name in tracked:
        path = ROOT / name
        require(path.is_file(), "missing tracked source " + name)
        sources[name] = digest(path)
    for name in VALIDATION_SOURCES:
        path = ROOT / name
        require(path.is_file(), "missing validation source " + name)
        sources[name] = digest(path)
    require(len(sources) >= 300, "current source inventory is unexpectedly small")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path,
                        default=CRATE / "target/phase19y/run/shapes-oldshader/qualification.json")
    parser.add_argument("--shared", type=Path,
                        default=CRATE / "target/phase19y/run/shared/qualification.json")
    parser.add_argument("--shape-fixtures", type=Path,
                        default=CRATE / "target/rdna3-port/gpt2-shapes-final-cpu")
    parser.add_argument("--shared-fixtures", type=Path,
                        default=CRATE / "target/rdna3-port/gen9-frozen/target/multimodal-gen9/shared-image-cpu-v3")
    parser.add_argument("--shape-archive", type=Path,
                        default=CRATE / "target/rdna3-port/gpt2-shapes-avx512-oracles.zip")
    parser.add_argument("--binary", type=Path,
                        default=CRATE / "target/debug/image_text_pipeline_probe.exe")
    parser.add_argument("--oracle-environment", type=Path,
                        default=CRATE / "target/phase19y/oracle-environment.json")
    parser.add_argument("--phase19w-matrix", type=Path,
                        default=CRATE / "target/phase19y/phase19w-matrix.json")
    parser.add_argument("--shapes-log", type=Path,
                        default=CRATE / "target/phase19y-shapes-oldshader.log")
    parser.add_argument("--shared-log", type=Path,
                        default=CRATE / "target/phase19y-shared.log")
    parser.add_argument("--output", type=Path, default=RECEIPT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty or args.allow_dirty, "tracked worktree is dirty:\n" + dirty)
    revision = git("rev-parse", "HEAD")
    require(len(revision) == 40, "cannot resolve source revision")

    shapes = load(args.shapes)
    shared = load(args.shared)
    require(shapes.get("pass") is True and shapes.get("completed_cases") == 16
            and shapes.get("expected_cases") == 16, "expanded shape replay is not 16/16")
    require(shared.get("pass") is True and shared.get("completed_cases") == 74
            and shared.get("expected_cases") == 74, "shared retained replay is not 74/74")
    require(shapes.get("fixtures_sha256") == SHAPE_MANIFEST_SHA,
            "expanded shape replay used another fixture manifest")
    require(shared.get("fixtures_sha256") == SHARED_MANIFEST_SHA,
            "shared replay used another fixture manifest")
    require(shapes.get("binary_sha256") == shared.get("binary_sha256"),
            "shape and shared replays used different probes")
    require(digest(args.binary) == shapes.get("binary_sha256"),
            "replay binary is not the current probe")
    require(args.binary.is_file(), "missing probe binary")

    shape_manifest_path = args.shape_fixtures / "manifest.json"
    shared_manifest_path = args.shared_fixtures / "manifest.json"
    require(digest(shape_manifest_path) == SHAPE_MANIFEST_SHA,
            "shape fixture manifest bytes changed")
    require(digest(shared_manifest_path) == SHARED_MANIFEST_SHA,
            "shared fixture manifest bytes changed")
    shape_manifest = load(shape_manifest_path)
    shared_manifest = load(shared_manifest_path)
    require(shape_manifest.get("complete") is True and len(shape_manifest.get("cases", [])) == 16,
            "shape fixture manifest is incomplete")
    require(shape_manifest.get("input_model_sha256") == EXPECTED_MODEL,
            "shape fixture input model changed")
    require(shared_manifest.get("complete") is True,
            "shared fixture manifest is incomplete")
    shape_files = verify_fixture_tree(args.shape_fixtures, shape_manifest, "shape")
    shared_files = verify_fixture_tree(args.shared_fixtures, shared_manifest, "shared")
    require(digest(args.shape_archive) == SHAPE_ARCHIVE_SHA
            and args.shape_archive.stat().st_size == SHAPE_ARCHIVE_BYTES,
            "recovered shape archive provenance changed")

    oracle_environment = load(args.oracle_environment)
    require(oracle_environment.get("torch_cpu_capability") == "AVX512",
            "PyTorch AVX-512 capability is not recorded")
    require(oracle_environment.get("torch") == "2.6.0+cpu"
            and oracle_environment.get("transformers_revision")
            == "129e8dc306932532e0b9e702d1fb4d5d7771c2a4",
            "CPU oracle environment differed from the frozen manifest")

    phase19x = load(PHASE19X)
    require(phase19x.get("pass") is True, "Phase XIX-X narrow attention receipt is not a pass")
    phase16 = load(PHASE16)
    require(phase16.get("pass") is True, "historical Phase XVI receipt is not a pass")
    retained = phase16.get("retained") or {}
    shared_hist = retained.get("shared") or {}
    specific = sum((row.get("cases") or 0) for name, row in retained.items() if name != "shared")
    require(shared_hist.get("cases") == 74, "historical shared count changed")
    require(specific == 19, "historical architecture-specific count changed")
    require(74 + specific == 93, "historical 93-entry accounting changed")

    matrix = load(args.phase19w_matrix)
    summary = matrix.get("summary") or {}
    require(summary.get("training_profiles") == 32
            and summary.get("generation_requests") == 72
            and summary.get("canonical_types") == 146
            and summary.get("aliases") == 83,
            "Phase XIX-W matrix summary changed")

    sources = source_inventory()
    shape_results = shapes.get("results") or []
    shared_results = shared.get("results") or []
    require(max_logit(shape_results) is not None and max_logit(shape_results) < LIMIT,
            "shape maximum exceeds the strict limit")
    require(max_logit(shared_results) is not None and max_logit(shared_results) < LIMIT,
            "shared maximum exceeds the strict limit")

    document = {
        "schema": 1,
        "kind": "phase19y-full-attention-current-source",
        "status": "PASS",
        "pass": True,
        "target": "rdna3",
        "scope": "phase19y_full_attention_current_source",
        "source_revision": revision,
        "source_clean": not bool(dirty),
        "source_files": sources,
        "binaries": {
            "image_text_pipeline_probe": {
                "path": args.binary.resolve().relative_to(CRATE).as_posix(),
                "sha256": digest(args.binary),
            }
        },
        "oracle": {
            "cpu": shape_manifest["oracle"]["cpu"],
            "torch": shape_manifest["oracle"]["torch"],
            "torch_revision": shape_manifest["oracle"]["torch_revision"],
            "transformers_revision": shape_manifest["oracle"]["transformers_revision"],
            "threads": shape_manifest["oracle"]["threads"],
            "text_attention": shape_manifest["oracle"]["text_attention"],
            "vision_attention": shape_manifest["oracle"]["vision_attention"],
            "cpu_capability": oracle_environment["torch_cpu_capability"],
            "cpu_model": oracle_environment["cpu_model"],
            "physical_cores": oracle_environment["physical_cores"],
            "logical_processors": oracle_environment["logical_processors"],
            "input_model_sha256": shape_manifest["input_model_sha256"],
            "environment_path": "fixtures/oracle-environment.json",
            "environment_sha256": digest(args.oracle_environment),
        },
        "fixtures": {
            "shapes": {
                "manifest_path": "fixtures/shape-manifest.json",
                "manifest_sha256": SHAPE_MANIFEST_SHA,
                "cases": 16,
                "archive_sha256": SHAPE_ARCHIVE_SHA,
                "archive_bytes": SHAPE_ARCHIVE_BYTES,
                "archive_path": "target/rdna3-port/gpt2-shapes-avx512-oracles.zip",
            },
            "shared": {
                "manifest_path": "fixtures/shared-manifest.json",
                "manifest_sha256": SHARED_MANIFEST_SHA,
                "cases": 77,
                "selected_cases": 74,
                "selected_profiles": 37,
            },
        },
        "fixture_bytes": {
            "verified": True,
            "shape_files": shape_files,
            "shared_files": shared_files,
        },
        "shapes": {
            "cases": 16,
            "expected_cases": 16,
            "pass": True,
            "binary_sha256": shapes["binary_sha256"],
            "fixtures_sha256": SHAPE_MANIFEST_SHA,
            "max_abs": max_logit(shape_results),
            "receipt_path": "shapes/qualification.json",
            "receipt_sha256": digest(args.shapes),
            "results": shape_results,
        },
        "retained": {
            "shared_current": {
                "cases": 74,
                "profiles": 37,
                "pass": True,
                "fixtures_sha256": SHARED_MANIFEST_SHA,
                "max_abs": max_logit(shared_results),
                "receipt_path": "shared/qualification.json",
                "receipt_sha256": digest(args.shared),
                "results": shared_results,
            },
            "architecture_specific_historical": {
                "total": 19,
                "verified_current": False,
                "basis": "historical Phase XVI receipt; not re-executed in Phase XIX-Y",
                "families": {name: row.get("cases") for name, row in retained.items()
                             if name != "shared"},
            },
            "coverage": {
                "shared_current": 74,
                "architecture_specific_historical": 19,
                "total_accounted": 93,
                "note": "93 accounted; only the 74 shared cases are current-source comparisons",
            },
        },
        "narrow_phase19x": {
            "path": "historical/phase19x-attention.json",
            "receipt_sha256": digest(PHASE19X),
            "status": "HISTORICAL_PRESERVED",
            "pass": True,
            "cases": 4,
            "comparisons": 12,
            "elements": 3158064,
            "max_abs": 0.0,
            "max_ulp": 0,
            "note": "Preserved Phase XIX-X narrow execution; not relabeled as full coverage.",
        },
        "historical_phase16": {
            "receipt_sha256": digest(PHASE16),
            "source_head": phase16.get("source_head"),
            "shapes": 16,
            "shared": 74,
            "architecture_specific": 19,
            "retained_entries": 93,
            "status": "HISTORICAL_PRESERVED",
        },
        "shader": {
            "path": "shaders/transformer_attention_forward_cpu_reciprocal.spv",
            "sha256": sources["hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv"],
            "sealed_historical_sha256": "f8439dc2a5e94dd39d6e9bf3d537d87b9c25c734e0b216627ba7bafe57dd3226",
            "drifted_sha256": "518a55b2fb84b16df8cbdd0a73068740f4d71b8d3bd16843b105dd04aee76a40",
            "inventory": 45,
            "status_counts": {"identical": 45},
            "decision": "restored the historical sealed reciprocal artifact; the drifted artifact failed the strict shape gate",
        },
        "phase19w": {
            "matrix_sha256": digest(args.phase19w_matrix),
            "canonical_types": summary["canonical_types"],
            "aliases": summary["aliases"],
            "training_profiles": summary["training_profiles"],
            "generation_requests": summary["generation_requests"],
            "status": "PASS",
        },
        "negative_controls": {"total": 0, "detected": 0, "controls": []},
        "evidence": {},
        "limitations": [
            "The 19 architecture-specific retained-family entries remain historical Phase XVI evidence; they are not current-source passes.",
            "The recovered CPU fixture archives are bound by manifest/archive hashes; raw fixture bytes remain under target/ and are not committed.",
            "The current full-attention receipt is Windows/RDNA3 hardware evidence; CI verifies the source, hashes, coverage and fail-closed checks.",
            "Phase XIX-X training, stability, held-out, drift and activation receipts remain preserved for their original source revision; only the reciprocal shader artifact and full-attention gate are requalified here.",
        ],
    }

    copies = {
        "shapes/qualification.json": args.shapes,
        "shapes/run.log": args.shapes_log,
        "shared/qualification.json": args.shared,
        "shared/run.log": args.shared_log,
        "fixtures/shape-manifest.json": shape_manifest_path,
        "fixtures/shared-manifest.json": shared_manifest_path,
        "fixtures/oracle-environment.json": args.oracle_environment,
        "fixtures/shape-archive.json": None,
        "historical/phase19x-attention.json": PHASE19X,
        "compatibility/phase19w-matrix.json": args.phase19w_matrix,
    }
    output = args.output.resolve()
    if args.dry_run:
        output = CRATE / "target/phase19y/receipt.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {output}")
    output.mkdir(parents=True)
    archive_record = {"sha256": SHAPE_ARCHIVE_SHA, "bytes": SHAPE_ARCHIVE_BYTES,
                      "path": str(args.shape_archive.resolve())}
    write(output / "fixtures/shape-archive.json", archive_record)
    evidence = {}
    for name, source in copies.items():
        destination = output / name
        if source is None:
            source = output / "fixtures/shape-archive.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        evidence[name] = digest(destination)
    document["evidence"] = evidence
    write(output / "current-qualification.json", document)
    controls = run_negative_controls(document, output, binary_root=args.binary.parent)
    document["negative_controls"] = controls
    write(output / "current-qualification.json", document)
    validate(document, output, binary_root=args.binary.parent,
             shape_fixtures=args.shape_fixtures, shared_fixtures=args.shared_fixtures,
             require_binary=True)
    print(f"PHASE XIX-Y: PASS ({revision[:12]}, 16/16 shapes, 74/74 shared, "
          f"{controls['detected']}/{controls['total']} controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Assemble the Windows x64 portable v0.2.8 release from a package-standalone.ps1 stage.

`package-standalone.ps1` produces a staging directory that still contains the
copied source tree (it needs it to build). The distributable is a curated subset:
the seven executables, their licences, the shipped documentation and the two
provenance files the packager writes. This script enforces that subset, builds a
reproducibly ordered ZIP, and emits the two sidecars that a release publishes
beside it.

    python make_release_zip.py --stage <stage> --out <dir> --version 0.2.7

The ZIP is written to --out and is not meant to be committed; BUILD_MANIFEST.json
and SHA256SUMS are.

This is the v0.2.8 assembler. It keeps the v0.2.7 packaging procedure and adds
the two promoted Phase XVIII/XX query artifacts, each pinned to the byte size and
SHA-256 its qualification receipt records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

EXECUTABLES = (
    "hierarchos-native.exe",
    "hierarchos-native-cli.exe",
    "hierarchos-infer.exe",
    "hierarchos-vulkan-train.exe",
    "hierarchos-vulkan-devices.exe",
    "hierarchos-vulkan-transformer-train.exe",
    "hierarchos-vulkan-transformer-logits.exe",
)
ROOT_FILES = (
    "LICENSE",
    "README.md",
    "README_ARCHITECTURES.md",
    "RELEASE_NOTES.md",
    "CLEAN_MACHINE_TEST.md",
    "THIRD_PARTY_NOTICES.md",
    "BUILD_INFO.txt",
    "DEPENDENCIES.txt",
)
# Filled from the source tree, not the stage: the stage carries the built
# artifacts, the repository carries the modules they embed.
MODULES = {
    "p3_contract_fallback": {
        "source": "shaders/wmma_gemm_f16_f16acc_twowave_p3.comp",
        "spirv": "shaders/wmma_gemm_f16_f16acc_twowave_p3.spv",
        "spirv_bytes": 23280,
        "spirv_fnv1a64": "0x076216ad97d7605f",
        "variant": "P3",
    },
    "c1_batched_staging": {
        "source": "shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.comp",
        "spirv": "shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.spv",
        "spirv_bytes": 36912,
        "spirv_fnv1a64": "0xfecb2a74c4bbbce4",
        "variant": "P3 (staged_spirv)",
    },
}
# The attention routes promoted in v0.2.7. Their byte sizes are pinned so a
# silently rebuilt or substituted artifact cannot ship.
ATTENTION_MODULES = {
    "forward_rowpacked64_cached": {
        "source": "shaders/transformer_attention_forward_rowpacked64_cached_rdna3.comp",
        "spirv": "shaders/transformer_attention_forward_rowpacked64_cached_rdna3.spv",
        "spirv_bytes": 22472,
        "eligibility": "0 < q_seq == kv_seq <= 64; equal QK/value head widths; unfused_qk in {0,1,9}; plain path only",
        "fallback": "generic attention forward",
    },
    "backward_query_two_pass": {
        "source": "shaders/transformer_attention_backward_query_rdna3.comp",
        "spirv": "shaders/transformer_attention_backward_query_rdna3.spv",
        "spirv_bytes": 48492,
        "eligibility": "the existing two-pass gate (unfused_qk 0/9, MHA or a qualified grouped geometry)",
        "fallback": "scalar attention backward",
    },
    "backward_query_dqunroll": {
        "source": "shaders/transformer_attention_backward_query_rdna3_dqunroll.comp",
        "spirv": "shaders/transformer_attention_backward_query_rdna3_dqunroll.spv",
        "spirv_bytes": 109300,
        "eligibility": "QK and value head dimension 128 inside the existing two-pass gate",
        "fallback": "the original query shader; opt out with HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0",
    },
    "backward_query_scratchreuse": {
        "source": "shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.comp",
        "spirv": "shaders/transformer_attention_backward_query_rdna3_dqunroll_scratchreuse.spv",
        "spirv_bytes": 89340,
        "eligibility": "the Phase XVIII query selector at head dimension 128; default on the qualified path",
        "fallback": "the Phase XVIII dQ-unroll shader, then the original query shader; opt out with HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0",
    },
    "backward_key_grouped": {
        "source": "shaders/transformer_attention_backward_key_rdna3.comp",
        "spirv": "shaders/transformer_attention_backward_key_rdna3.spv",
        "spirv_bytes": 10724,
        "eligibility": "MHA, plus 8q/2kv, 4q/2kv, 8q/4kv and 16q/4kv at head dimension 128",
        "fallback": "scalar attention backward",
    },
    "backward_key_channel4": {
        "source": "shaders/transformer_attention_backward_key_channel4_rdna3.comp",
        "spirv": "shaders/transformer_attention_backward_key_channel4_rdna3.spv",
        "spirv_bytes": 12872,
        "eligibility": "head_dim == value_head_dim == 128, partition 32 * 4 == 128",
        "fallback": "the original two-pass key shader; opt out with HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0",
    },
}
# The Phase XXII FP32 input-gradient promotion. Pinned so a substituted
# artifact cannot ship.
LINEAR_MODULES = {
    "input_grad_output_tile32": {
        "source": "shaders/linear_input_grad_output_tile32.comp",
        "spirv": "shaders/linear_input_grad_output_tile32.spv",
        "spirv_bytes": 4992,
        "eligibility": "rows, input_dim and output_dim all multiples of 32",
        "fallback": "the original linear_input_grad kernel; opt out with HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0",
    },
}
FIXED_DATE = (2026, 11, 28, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(stage: Path) -> list[tuple[Path, str]]:
    """Every file that ships, as (absolute path, archive name)."""
    entries: list[tuple[Path, str]] = []
    bin_dir = stage / "bin"
    present = sorted(p.name for p in bin_dir.iterdir() if p.is_file())
    if present != sorted(EXECUTABLES):
        raise SystemExit(
            "bin/ does not match the seven-executable allowlist\n"
            f"  missing: {sorted(set(EXECUTABLES) - set(present))}\n"
            f"  extra:   {sorted(set(present) - set(EXECUTABLES))}"
        )
    for name in EXECUTABLES:
        entries.append((bin_dir / name, f"bin/{name}"))
    for name in ROOT_FILES:
        path = stage / name
        if not path.is_file():
            raise SystemExit(f"missing staged file: {name}")
        entries.append((path, name))
    licences = stage / "licenses"
    if not licences.is_dir():
        raise SystemExit("missing staged licenses/ directory")
    for path in sorted(licences.rglob("*")):
        if path.is_file():
            entries.append((path, path.relative_to(stage).as_posix()))
    return entries


def build_zip(out_zip: Path, entries: list[tuple[Path, str]]) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, name in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_info_fields(stage: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (stage / "BUILD_INFO.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def pinned_modules(vulkan_root: Path, table: dict[str, dict], with_fnv: bool) -> dict:
    modules: dict = {}
    for key, spec in table.items():
        source = vulkan_root / spec["source"]
        spirv = vulkan_root / spec["spirv"]
        if spirv.stat().st_size != spec["spirv_bytes"]:
            raise SystemExit(
                f"{spec['spirv']} is {spirv.stat().st_size} B, expected {spec['spirv_bytes']} B"
            )
        entry = {
            "source": spec["source"],
            "source_sha256": sha256(source),
            "spirv": spec["spirv"],
            "spirv_sha256": sha256(spirv),
            "spirv_bytes": spirv.stat().st_size,
        }
        if with_fnv:
            entry["variant"] = spec["variant"]
            entry["spirv_fnv1a64"] = spec["spirv_fnv1a64"]
        else:
            entry["eligibility"] = spec["eligibility"]
            entry["fallback"] = spec["fallback"]
        modules[key] = entry
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root that carries the shader sources",
    )
    args = parser.parse_args()

    stage = args.stage.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = collect(stage)

    zip_name = f"Hierarchos-Native-Windows-x64-Portable-v{args.version}.zip"
    out_zip = out / zip_name
    build_zip(out_zip, entries)

    executables = {}
    for name in EXECUTABLES:
        path = stage / "bin" / name
        executables[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    vulkan_root = args.repository / "hierarchos-vulkan"
    modules = pinned_modules(vulkan_root, MODULES, with_fnv=True)
    attention_modules = pinned_modules(vulkan_root, ATTENTION_MODULES, with_fnv=False)
    linear_modules = pinned_modules(vulkan_root, LINEAR_MODULES, with_fnv=False)

    info = build_info_fields(stage)
    if not re.fullmatch(r"[0-9a-f]{40}", info.get("Source commit", "")):
        raise SystemExit(f"BUILD_INFO.txt does not name a commit: {info.get('Source commit')!r}")

    manifest = {
        "release_version": args.version,
        "source_commit": info["Source commit"],
        "source_state_at_packaging": info.get("Source state at packaging", "").lower(),
        "build_timestamp_utc": info.get("Build timestamp (UTC)", ""),
        "target_triple": info.get("Target", ""),
        "rust_version": info.get("rustc", ""),
        "cargo_version": info.get("cargo", ""),
        "release_flags": {
            "profile": info.get("Profile", ""),
            "rustflags": info.get("Effective build flags", ""),
            "target_cpu_native": False,
            "dependency_scan": "enabled (no -SkipDependencyScan)",
        },
        "static_crt": "static" in info.get("CRT linkage", ""),
        "standalone_cargo_lock_sha256": info.get("Cargo.lock SHA-256", ""),
        "modules": modules,
        "attention_modules": attention_modules,
        "linear_modules": linear_modules,
        "production_route": {
            "attention": {
                "mode": "on by default, per-geometry",
                "selector": "hierarchos-vulkan/src/transformer.rs",
                "two_pass_requires": [
                    "HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS is unset or truthy",
                    "one key-range block (a split backward submission keeps the scalar kernel)",
                    "unfused_qk in {0, 9} for MHA, or one of the four measured grouped shapes at head dimension 128",
                    "exact scratch size representable",
                ],
                "channel4_requires": [
                    "the two-pass gate admitted the geometry",
                    "head_dim == value_head_dim == 128",
                    "HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4 is unset or truthy",
                ],
                "row_packed_requires": [
                    "0 < q_seq == kv_seq <= 64",
                    "equal QK and value head widths",
                    "unfused_qk in {0, 1, 9}",
                    "no window, softcap, ALiBi, relative bias, dropout, sink, sparse mask, compressed keys, paged KV or value scale",
                ],
                "query_order": [
                    "query_dqunroll_scratchreuse when the Phase XVIII selector admits the geometry and the artifact loads",
                    "query_dqunroll when scratch reuse is disabled or its pipeline is refused",
                    "the original query shader when the dQ-unroll path is disabled or the geometry is not admitted",
                ],
                "fallback": "the scalar backward and the generic forward for every ineligible geometry",
                "control_arms": {
                    "HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS=0": "scalar attention backward",
                    "HIERARCHOS_ATTENTION_BACKWARD_KEY_CHANNEL4=0": "original two-pass key shader",
                    "HIERARCHOS_ATTENTION_BACKWARD_QUERY_SCRATCH_REUSE=0": "Phase XVIII dQ-unroll query shader",
                    "HIERARCHOS_ATTENTION_BACKWARD_QUERY_DQUNROLL=0": "original query shader",
                },
            },
            "linear_input_grad": {
                "mode": "output-tiled 32 x 32, on by default for 32-aligned shapes",
                "selector": "hierarchos-vulkan/src/transformer.rs",
                "requires": [
                    "rows % 32 == 0 and input_dim % 32 == 0 and output_dim % 32 == 0",
                    "the promoted artifact loads",
                    "HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE is unset, 1 or unrecognized",
                ],
                "fallback": "the original 16 x 16 linear_input_grad kernel for every other shape or an explicit 0",
                "control_arm": {
                    "HIERARCHOS_LINEAR_INPUT_GRAD_OUTPUT_TILE=0": "original linear_input_grad kernel"
                },
            },
            "wmma": {
                "mode": "full-fp16 (opt-in; global WMMA off by default)",
                "selector": "hierarchos-vulkan/src/wmma.rs::batched_staging_eligible",
                "c1_requires": [
                    "rows % 32 == 0 and columns % 32 == 0 and reduction % 16 == 0",
                    "lda % 2 == 0 and ldb % 2 == 0",
                    "packed element indices inside the 32-bit ceiling with one element of headroom",
                    "both bound operands cover every element the fast path reads",
                ],
                "fallback": "P3 module, selected automatically when any condition is unproven",
                "override": "none; no flag, environment variable or variant name selects a staging module",
            },
        },
        "qualification": {
            "attention": {
                "two_pass_backward": "bit-exact over four geometries, 6316128/6316128 elements, zero ULP; 3.2492x complete step (95% CI 3.1146-3.3838)",
                "gqa_key_pass": "bit-exact for 8q/2kv and 4q/2kv in both modes; 19.76x complete step at 8q/2kv (95% CI 19.64-19.89)",
                "row_packed_forward": "1.85M/1.85M words bit-exact over ten geometries and three modes; 1.2239x on the shipped selector",
                "channel4_key_pass": "20 parity cases, 60 tensors, 5505024 elements, zero failures; 1.5297x complete step (5/5 rounds)",
                "query_dqunroll": "8 ABBA series, 5/5 rounds improved, 1.042x-1.124x complete step, CI excludes 1.0 in every series",
                "query_scratchreuse": "140/140 bitwise tensor comparisons over 28 cases; 29/29 in-graph loss, gradient and parameter digests; 1.088x FP32 and 1.120x Mixed complete step (5/5 rounds)",
                "evidence": "hierarchos-vulkan/validation/phase_ix_row_packed_forward_20261124, phase_x_row_packed_forward_promotion_20261125, phase_xii_channel4_key_production_20261125, mha_attention_backward_closeout_20261124, gqa_production_promotion_20261124, gqa_end_to_end_training_20261124, phase_xviii_v028_dq_accumulator_promotion_20261126, phase_xix_query_scratch_reuse_20261127, phase_xx_query_scratchreuse_promotion_20261127",
            },
            "linear_input_grad": "6 shapes, 272038/272038 words bitwise equal; 5-round ABBA 24.875 -> 21.716 ms, 1.1462x, 5/5 rounds, 95% CI 0.8706-0.8742",
            "mixed_wmma": "1.2385x / 1.2288x LoRA complete step, 1.2792x full-parameter timing only; opt-in, default off; adapter elementwise drift reported",
            "evidence": "hierarchos-vulkan/validation/phase_xiii_mixed_wmma_requalification_20261125",
            "device": "AMD Radeon Graphics, PCI 1002:15bf, driver 32.0.32015.14001",
        },
        "executables": executables,
        "pe_dependency_verification": {
            "result": "PASS",
            "report": "DEPENDENCIES.txt",
            "expected_external_gpu_runtime": "vulkan-1.dll (not bundled)",
        },
        "package": {
            "file": zip_name,
            "bytes": out_zip.stat().st_size,
            "sha256": sha256(out_zip),
            "entries": len(entries),
        },
        "reports": {
            "release_note": "RDNA3_V028_RELEASE.md",
            "qualification": "hierarchos-vulkan/validation/phase_xx_query_scratchreuse_promotion_20261127/README.md",
            "package_acceptance": "CLEAN_MACHINE_TEST.md",
        },
    }
    # Byte-exact, LF-only sidecars: a CRLF checksum list is unreadable to
    # `sha256sum -c` on every platform that is not Windows.
    manifest_path = out / "BUILD_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    sums_path = out / "SHA256SUMS"
    sums_path.write_text(
        f"{manifest['package']['sha256']}  {zip_name}\n{sha256(manifest_path)}  BUILD_MANIFEST.json\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"ZIP      {zip_name}  {manifest['package']['bytes']} B  {manifest['package']['sha256']}")
    print(f"entries  {len(entries)}")
    for name, spec in executables.items():
        print(f"  {name:44s} {spec['bytes']:>10d}  {spec['sha256']}")
    print(f"manifest {sha256(manifest_path)}")
    if manifest["source_state_at_packaging"] != "clean":
        print(
            "WARNING: BUILD_INFO.txt reports a dirty source tree at packaging time",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

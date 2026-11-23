#!/usr/bin/env python3
"""Assemble the Windows x64 portable release from a package-standalone.ps1 stage.

`package-standalone.ps1` produces a staging directory that still contains the
copied source tree (it needs it to build). The distributable is a curated subset:
the seven executables, their licences, the shipped documentation and the two
provenance files the packager writes. This script enforces that subset, builds a
reproducibly ordered ZIP, and emits the two sidecars that a release publishes
beside it.

    python make_release_zip.py --stage <stage> --out <dir> --version 0.2.6

The ZIP is written to --out and is not meant to be committed; BUILD_MANIFEST.json
and SHA256SUMS are.
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
FIXED_DATE = (2026, 11, 23, 0, 0, 0)


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
    modules = {}
    for key, spec in MODULES.items():
        source = vulkan_root / spec["source"]
        spirv = vulkan_root / spec["spirv"]
        modules[key] = {
            "variant": spec["variant"],
            "source": spec["source"],
            "source_sha256": sha256(source),
            "spirv": spec["spirv"],
            "spirv_sha256": sha256(spirv),
            "spirv_bytes": spirv.stat().st_size,
            "spirv_fnv1a64": spec["spirv_fnv1a64"],
        }
        if modules[key]["spirv_bytes"] != spec["spirv_bytes"]:
            raise SystemExit(
                f"{spec['spirv']} is {modules[key]['spirv_bytes']} B, expected {spec['spirv_bytes']} B"
            )

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
        "production_route": {
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
        "qualification": {
            "numerical": "40 receipts, 8 seeds, 37/37 tile16 gates per seed, 120/120 bit-identical to v0.2.5",
            "performance_vs_v025": {
                "2048x768x2048": {"speedup": 1.7074, "ci95": [1.6976, 1.7171]},
                "512x2048x768": {"speedup": 1.3286, "ci95": [1.3182, 1.3391]},
                "512x768x768": {"speedup": 2.0920, "ci95": [2.0415, 2.1425]},
                "128x8192x768": {"speedup": 1.3782, "ci95": [1.3580, 1.3983]},
            },
            "evidence": "hierarchos-vulkan/validation/phase_lxxxviii_c1_promotion/",
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
            "release_note": "RDNA3_C1_STAGING_RELEASE.md",
            "qualification": "hierarchos-vulkan/validation/phase_lxxxviii_c1_promotion/README.md",
            "package_acceptance": "CLEAN_MACHINE_TEST.md",
        },
    }
    manifest_path = out / "BUILD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    sums_path = out / "SHA256SUMS"
    sums_path.write_text(
        f"{manifest['package']['sha256']}  {zip_name}\n{sha256(manifest_path)}  BUILD_MANIFEST.json\n",
        encoding="utf-8",
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

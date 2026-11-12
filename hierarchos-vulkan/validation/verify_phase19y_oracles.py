#!/usr/bin/env python3
"""Verify the immutable Phase XIX-Y CPU oracle archives byte-for-byte.

The archives are published as GitHub Release assets.  CI downloads them by
tag, verifies the archive hash and size before opening the ZIP, extracts only
safe relative entries into a temporary tree, and then checks every file named
by the frozen manifests.  The extracted trees are passed to the Phase XIX-Y
checker so the receipt is bound to real bytes rather than a JSON claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

CRATE = pathlib.Path(__file__).resolve().parents[1]

SHAPE_ARCHIVE_SHA = "a45396b752d62b01f75aa0f0cc8a83ee71640bf44a9f83b841e7476056b6ef14"
SHAPE_ARCHIVE_BYTES = 32948229
SHAPE_MANIFEST_SHA = "709427881624237c05dccf1e9bdd7593285819304af7d3778b885bc206fbad0c"

SHARED_ARCHIVE_SHA = "8a85120ef2a27f9567788c39f4668c6dc52c808e8cae2c179ad8093119d0a95c"
SHARED_ARCHIVE_BYTES = 26077578
SHARED_MANIFEST_SHA = "a98dc5c32f4ba044b59a7a1e03fcac2b32d234ae12fde7af8ca3b1ff00cad1be"

MAX_ENTRIES = 5000
MAX_UNCOMPRESSED = 256 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_name(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    parts = pathlib.PurePosixPath(name).parts
    return not any(part in ("", ".", "..") for part in parts)


def verify_archive(archive: pathlib.Path, expected_sha: str, expected_bytes: int,
                   label: str, destination: pathlib.Path) -> set[str]:
    """Check the archive before extraction, then extract only safe entries."""
    require(archive.is_file(), f"missing {label} archive: {archive}")
    require(archive.stat().st_size == expected_bytes,
            f"{label} archive size changed: {archive.stat().st_size}")
    require(digest(archive) == expected_sha, f"{label} archive hash changed")

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        require(len(infos) <= MAX_ENTRIES, f"{label} archive has too many entries")
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), f"{label} archive has duplicate entries")
        total = 0
        for info in infos:
            require(_safe_name(info.filename),
                    f"{label} archive has an unsafe path: {info.filename!r}")
            require(not info.is_dir(), f"{label} archive contains a directory entry")
            require(not (info.flag_bits & 0x1), f"{label} archive is encrypted")
            require(((info.external_attr >> 16) & 0o170000) != 0o120000,
                    f"{label} archive contains a symlink")
            total += info.file_size
        require(total <= MAX_UNCOMPRESSED, f"{label} archive expands beyond the limit")

        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        root = destination.resolve()
        for info in infos:
            target = (destination / info.filename).resolve()
            require(root in target.parents or target.parent == root,
                    f"{label} archive escapes extraction root: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return {info.filename for info in infos}


def verify_manifest(root: pathlib.Path, expected_manifest_sha: str,
                    label: str) -> dict:
    """Require exact file-set equality and re-hash every manifest-listed file."""
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing {label} manifest")
    require(digest(manifest_path) == expected_manifest_sha,
            f"{label} manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("complete") is True, f"{label} manifest is incomplete")
    cases = manifest.get("cases") or []
    require(cases, f"{label} manifest contains no cases")

    case_files = {
        relative: expected
        for case in cases
        for relative, expected in (case.get("files") or {}).items()
    }
    oracle_files = (manifest.get("oracle") or {}).get("source_sha256") or {}
    expected = set(case_files) | set(oracle_files) | {"manifest.json"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == expected,
            f"{label} fixture file set changed: "
            f"missing={sorted(expected - actual)[:3]} extra={sorted(actual - expected)[:3]}")

    for relative, expected in case_files.items():
        require(digest(root / relative) == expected,
                f"altered {label} fixture: {relative}")
    for relative, expected in oracle_files.items():
        require(digest(root / relative) == expected,
                f"altered {label} oracle source: {relative}")
    if label == "shape":
        require(manifest.get("input_model_sha256"), f"{label} input model is unbound")
    return {
        "manifest_sha256": expected_manifest_sha,
        "cases": len(cases),
        "case_files": len(case_files),
        "oracle_source_files": len(oracle_files),
        "total_files": len(actual),
        "manifest": manifest,
    }


def run_checker(shape_root: pathlib.Path, shared_root: pathlib.Path,
                receipt: pathlib.Path) -> str:
    checker = CRATE / "validation/check_phase19y_full_attention.py"
    result = subprocess.run(
        [sys.executable, str(checker), "--receipt", str(receipt),
         "--shape-fixtures", str(shape_root),
         "--shared-fixtures", str(shared_root)],
        cwd=CRATE, capture_output=True, text=True,
    )
    output = (result.stdout + result.stderr).strip()
    require(result.returncode == 0, "Phase XIX-Y checker rejected extracted fixtures: " + output[-600:])
    return output


def verify(shape_archive: pathlib.Path, shared_archive: pathlib.Path,
           work: pathlib.Path, receipt: pathlib.Path) -> dict:
    shape_root = work / "shape"
    shared_root = work / "shared"
    shape_entries = verify_archive(
        shape_archive, SHAPE_ARCHIVE_SHA, SHAPE_ARCHIVE_BYTES, "shape", shape_root
    )
    require(len(shape_entries) == 363, "shape archive entry count changed")
    shared_entries = verify_archive(
        shared_archive, SHARED_ARCHIVE_SHA, SHARED_ARCHIVE_BYTES, "shared", shared_root
    )
    require(len(shared_entries) == 1119, "shared archive entry count changed")

    shape = verify_manifest(shape_root, SHAPE_MANIFEST_SHA, "shape")
    shared = verify_manifest(shared_root, SHARED_MANIFEST_SHA, "shared")
    checker_output = run_checker(shape_root, shared_root, receipt)
    return {
        "schema": 1,
        "status": "PASS",
        "shape": {
            "archive_sha256": SHAPE_ARCHIVE_SHA,
            "archive_bytes": SHAPE_ARCHIVE_BYTES,
            "entries": len(shape_entries),
            "cases": shape["cases"],
            "case_files": shape["case_files"],
            "oracle_source_files": shape["oracle_source_files"],
            "total_files": shape["total_files"],
        },
        "shared": {
            "archive_sha256": SHARED_ARCHIVE_SHA,
            "archive_bytes": SHARED_ARCHIVE_BYTES,
            "entries": len(shared_entries),
            "cases": shared["cases"],
            "case_files": shared["case_files"],
            "oracle_source_files": shared["oracle_source_files"],
            "total_files": shared["total_files"],
        },
        "checker": checker_output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-archive", type=pathlib.Path, required=True)
    parser.add_argument("--shared-archive", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path,
                        default=CRATE / "validation/release_parity/multimodal_training"
                        / "20261112-phase19y-full-attention/current-qualification.json")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    work = args.work or pathlib.Path(tempfile.mkdtemp(prefix="phase19y-oracles-", dir=CRATE / "target"))
    try:
        result = verify(args.shape_archive, args.shared_archive, work, args.receipt)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"PHASE XIX-Y ORACLES: FAIL ({error})")
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PHASE XIX-Y ORACLES: PASS "
          f"({result['shape']['case_files']} shape + {result['shared']['case_files']} shared "
          f"case files; {result['shape']['total_files']} + {result['shared']['total_files']} total files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

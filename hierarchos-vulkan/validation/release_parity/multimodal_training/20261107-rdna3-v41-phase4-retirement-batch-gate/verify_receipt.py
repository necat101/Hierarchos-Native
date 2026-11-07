#!/usr/bin/env python3
"""Verify every file bound by this receipt's manifest.json."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        print("FAIL: manifest has no files")
        return 1
    failures = []
    for relative, expected in sorted(files.items()):
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif digest(path) != expected:
            failures.append(f"hash mismatch {relative}")
    extra = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in files
        and path.name != "manifest.json"
    )
    if extra:
        failures.append(f"unbound files: {', '.join(extra)}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print(f"PASS: {len(files)} files match manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

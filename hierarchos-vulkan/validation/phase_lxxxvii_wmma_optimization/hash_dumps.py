#!/usr/bin/env python3
"""Pin the Phase LXXXVII production-dump bytes without committing them.

The production dumps are ~16 MB per seed per arm, so they stay outside Git. This
script records each file's byte count and SHA-256 in exactly the shape Phase
LXXXVI used (`dumps_manifest.json`), so a verifier can tell whether a file on
disk is the file this record was written from.

Usage: hash_dumps.py <out.json> <arm-dir> [<arm-dir> ...]

Each `<arm-dir>` must contain a `dumps/` subdirectory. The arm label is the
directory name.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    out = Path(sys.argv[1])
    arms = [Path(arg) for arg in sys.argv[2:]]
    manifest: dict[str, dict[str, dict]] = {}
    for arm in arms:
        dumps = arm / "dumps"
        if not dumps.is_dir():
            raise SystemExit(f"no dumps directory in {arm}")
        entries = {}
        for path in sorted(dumps.glob("dump_prod_p3_*.json")):
            entries[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        if not entries:
            raise SystemExit(f"no production dumps in {dumps}")
        manifest[arm.name] = entries
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(entry["bytes"] for arm in manifest.values() for entry in arm.values())
    print(f"{sum(len(a) for a in manifest.values())} dumps, {total} bytes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

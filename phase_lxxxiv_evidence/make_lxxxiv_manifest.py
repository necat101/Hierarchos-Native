#!/usr/bin/env python3
"""Independent SHA-256 manifest for the Phase LXXXIV evidence.

GNU `sha256sum` format (`<digest>  <path>`), paths relative to this directory,
sorted, and excluding the manifest itself. Nothing here is derived from a
receipt: it hashes whatever bytes are on disk, so it can only reproduce when the
committed evidence is byte-exact.

Usage:
  make_lxxxiv_manifest.py            # write SHA256SUMS_phase_lxxxiv.txt
  make_lxxxiv_manifest.py --check    # verify the committed manifest
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SHA256SUMS_phase_lxxxiv.txt"
# Regenerated caches, the raw operand dumps and the manifest itself are not
# committed evidence. The dumps are pinned by digest in `dumps_index.json`
# instead of being stored: Phase LXXIX kept its 386 MB dump tree out of Git, and
# the tensors are regenerable from the archived harness with `--dump-operands`.
SKIP_DIRS = {"__pycache__", ".pytest_cache", "dumps"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def inventory() -> list[Path]:
    out = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(HERE)
        if relative == Path(MANIFEST.name):
            continue
        if SKIP_DIRS & set(relative.parts):
            continue
        out.append(relative)
    return out


def main() -> int:
    entries = [(digest(HERE / rel), rel.as_posix()) for rel in inventory()]
    if "--check" in sys.argv[1:]:
        if not MANIFEST.is_file():
            print("manifest missing")
            return 1
        recorded = {}
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sha, name = line.split("  ", 1)
            recorded[name] = sha
        current = dict((name, sha) for sha, name in entries)
        missing = sorted(set(recorded) - set(current))
        extra = sorted(set(current) - set(recorded))
        changed = sorted(n for n in set(recorded) & set(current) if recorded[n] != current[n])
        print(f"entries {len(entries)}  recorded {len(recorded)}")
        if missing:
            print("missing:", missing[:5])
        if extra:
            print("extra:", extra[:5])
        if changed:
            print("changed:", changed[:5])
        ok = not (missing or extra or changed)
        print("RESULT", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    MANIFEST.write_text(
        "".join(f"{sha}  {name}\n" for sha, name in entries), encoding="utf-8"
    )
    print(f"wrote {MANIFEST.name}: {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

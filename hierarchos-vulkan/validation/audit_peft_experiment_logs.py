"""Inventory preserved experiment logs without treating historical passes as current."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-09-23")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()
    rows = []
    interesting = re.compile(r'(?i)(error(?:\[|:)|failed|failure|panic|traceback|regress|test result:|"pass"\s*:|"(?:[ab]_)?(?:gradient|base|two_step_adamw|peft|frozen_base|resume)_max_abs"\s*:)')
    paths = []
    for directory, children, files in os.walk(ROOT.parent):
        children[:] = [name for name in children if name not in {"target", ".git", "node_modules"}]
        paths.extend(Path(directory) / name for name in files if name.endswith(".log"))
    for path in sorted(paths):
        stat = path.stat()
        if stat.st_mtime < since:
            continue
        data = path.read_bytes()
        # PowerShell redirection in older sessions produced UTF-16 logs.
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        lines = data.decode(encoding, errors="replace").splitlines()
        matches = [{"line": i, "text": line[:2000]} for i, line in enumerate(lines, 1) if interesting.search(line)]
        rows.append({"path": str(path.relative_to(ROOT.parent)),
                     "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "line_count": len(lines), "evidence": matches,
                     "tail": lines[-8:]})
    report = {"since_utc": args.since, "created_utc": datetime.now(timezone.utc).isoformat(),
              "qualification": "Historical evidence only; modification times do not establish execution provenance.",
              "logs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"logs": len(rows), "with_failure_text": sum(any(re.search(r'(?i)(error:|failed|panic|traceback|"pass": false)', x['text']) for x in row['evidence']) for row in rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()

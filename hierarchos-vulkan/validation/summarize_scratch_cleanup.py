"""Group the scratch-cleanup ledger by artifact class.

The ledger records one entry per removed root; the summary that lands in the
audit needs the byte count per *class* of artifact, so this derives that from
the ledger rather than from a truncated console listing.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "release_parity"
    / "deepseek_v41"
    / "scratch-cleanup.json"
)


def classify(path: str) -> str:
    normalized = path.replace("\\", "/")
    if "Temp/" in normalized or "/tmp/" in normalized:
        return "os temp scratch"
    if ".peft-oracle-fixtures" in normalized:
        return "superseded oracle priors"
    if ".peft-" in normalized:
        return "stale peft work scratch"
    if "/fixtures/" in normalized:
        return "regenerable generated packages"
    return "other"


LABELS = {
    "regenerable_generated_package": "regenerable generated packages",
    "stale_peft_work_scratch": "stale peft work scratch",
    "os_temp_scratch": "os temp scratch",
}


def main(argv: list[str]) -> int:
    ledger = json.loads(Path(argv[1] if len(argv) > 1 else DEFAULT).read_text())
    runs = ledger.get("runs") or [ledger]
    totals: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    bytes_reported = 0
    files_reported = 0
    for run in runs:
        bytes_reported += run["deleted_bytes"]
        files_reported += run["deleted_files"]
        if not run["roots"]:
            # A pass that predates per-root history is recorded by aggregate.
            key = run.get("aggregate_only", "unclassified")
            totals[key] += run["deleted_bytes"]
            counts[key] += run["deleted_files"]
        for root in run["roots"]:
            key = LABELS.get(root.get("class", ""), None) or classify(root["path"])
            totals[key] += root["bytes_removed"]
            counts[key] += root["files_removed"]
    print(f"method: {ledger['method']}")
    print(f"runs: {len(runs)}")
    for key, value in totals.most_common():
        print(f"{key:34s} {value:>15,} bytes {counts[key]:>7,} files")
    print(f"{'sum':34s} {sum(totals.values()):>15,} bytes {sum(counts.values()):>7,} files")
    if sum(totals.values()) != bytes_reported:
        # Path-derived classes are a view of the same deletions, so any
        # disagreement means the ledger itself is inconsistent.
        raise SystemExit(
            f"class totals {sum(totals.values()):,} disagree with reported {bytes_reported:,}"
        )
    print(f"reported: {bytes_reported:,} bytes in {files_reported:,} files")
    first, last = runs[0], runs[-1]
    print(
        f"free {first['started_free_bytes']:,} -> {last['finished_free_bytes']:,}  "
        f"delta {last['finished_free_bytes'] - first['started_free_bytes']:,}"
    )
    print(f"retained {ledger['retained_scalable_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Restore the first reclaim run into the now-accumulating cleanup ledger.

The first pass wrote its ledger before the ledger recorded history, so the second
pass overwrote it. The first pass's per-root byte counts were captured in its
console output and are re-entered here; their sum is checked against the total
that pass reported, so the reconstruction cannot silently drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "hierarchos-vulkan" / "validation" / "fixtures"
LEDGER = (
    REPO
    / "hierarchos-vulkan"
    / "validation"
    / "release_parity"
    / "deepseek_v41"
    / "scratch-cleanup.json"
)

REPORTED_FIRST_RUN_BYTES = 16_950_908_454
REPORTED_FIRST_RUN_FILES = 44_293
FIRST_RUN_FREE_BEFORE = 41_334_362_112
FIRST_RUN_FREE_AFTER = 58_423_398_400

# The second pass removed superseded oracle priors only. Its per-root detail
# predates the history feature, and the ledger's top-level fields have since been
# overwritten by later passes, so its captured totals are pinned here too.
SECOND_RUN_BYTES = 46_177_687
SECOND_RUN_FILES = 5_304
SECOND_RUN_FREE_BEFORE = 58_423_398_400
SECOND_RUN_FREE_AFTER = 58_484_158_464

# (relative or absolute path, bytes removed, files removed) exactly as the first
# pass printed them.
PACKAGES = [
    ("deepseek_v41_production_4141090", 8_342_133_789, 22),
    ("deepseek_v41_lora_4141095", 252_520_823, 5),
    ("deepseek_v41_wide_4141071", 3_138_259, 2),
    ("deepseek_v41_wide_4141072", 17_435_602, 4),
    ("deepseek_v41_wide_4141073", 3_155_047, 4),
    ("deepseek_v41_structure_4141101", 7_856_379, 4),
    ("deepseek_v41_structure_4141102", 3_047_688, 4),
    ("deepseek_v41_schedule_width_4141211", 763_677, 3),
]
PEFT_WORK = [
    ("hierarchos-vulkan/.peft-bank-work", 187_464_068, 19_181),
    ("hierarchos-vulkan/.peft-validation-work", 173_656_802, 24_438),
]
# 22 packed scratch directories at 166,177,280 bytes, one at 332,354,560 (two
# tensors) and one empty; two vocabulary directories.
TEMP_PACKED = (22, 166_177_280, 332_354_560, 23)
TEMP_VOCABULARY = [2_647_654_400, 1_323_827_200]


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    roots: list[dict[str, object]] = []

    for name, size, files in PACKAGES:
        roots.append(
            {
                "path": str(FIXTURES / name),
                "allowed_root": str(FIXTURES),
                "files_removed": files,
                "bytes_removed": size,
                "class": "regenerable_generated_package",
            }
        )
    for name, size, files in PEFT_WORK:
        roots.append(
            {
                "path": str(REPO / name),
                "allowed_root": str((REPO / name).parent),
                "files_removed": files,
                "bytes_removed": size,
                "class": "stale_peft_work_scratch",
            }
        )
    count, each, doubled, directories = TEMP_PACKED
    for index in range(count):
        roots.append(
            {
                "path": f"<temp>/deepseek-v41-packed-{index:02d}",
                "allowed_root": "<temp>",
                "files_removed": 26,
                "bytes_removed": each,
                "class": "os_temp_scratch",
            }
        )
    roots.append(
        {
            "path": "<temp>/deepseek-v41-packed-58ejc3_n",
            "allowed_root": "<temp>",
            "files_removed": 52,
            "bytes_removed": doubled,
            "class": "os_temp_scratch",
        }
    )
    for index, size in enumerate(TEMP_VOCABULARY):
        roots.append(
            {
                "path": f"<temp>/deepseek-v41-vocabulary-{index:02d}",
                "allowed_root": "<temp>",
                "files_removed": 1,
                "bytes_removed": size,
                "class": "os_temp_scratch",
            }
        )

    total = sum(root["bytes_removed"] for root in roots)
    files = sum(root["files_removed"] for root in roots)
    if total != REPORTED_FIRST_RUN_BYTES or files != REPORTED_FIRST_RUN_FILES:
        raise SystemExit(
            f"reconstruction disagrees with the reported first pass: "
            f"{total} vs {REPORTED_FIRST_RUN_BYTES} bytes, "
            f"{files} vs {REPORTED_FIRST_RUN_FILES} files"
        )

    first_run = {
        "started_free_bytes": FIRST_RUN_FREE_BEFORE,
        "finished_free_bytes": FIRST_RUN_FREE_AFTER,
        "deleted_bytes": total,
        "deleted_files": files,
        "roots": roots,
    }
    second_run = {
        "started_free_bytes": SECOND_RUN_FREE_BEFORE,
        "finished_free_bytes": SECOND_RUN_FREE_AFTER,
        "deleted_bytes": SECOND_RUN_BYTES,
        "deleted_files": SECOND_RUN_FILES,
        "roots": [],
        "aggregate_only": "superseded oracle priors",
    }
    # Normalise the two reconstructed head entries and keep every later pass, so
    # running this repeatedly is idempotent and never resets the record.
    runs = ledger.get("runs", [])
    reconstructed_head = bool(runs) and runs[0].get("roots") == roots
    tail = runs[2:] if reconstructed_head else runs
    ledger["runs"] = [first_run, second_run] + tail
    ledger["reconciled_first_run"] = True
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(
        f"recorded {len(ledger['runs'])} runs; first {total:,} bytes / {files:,} files, "
        f"second {second_run['deleted_bytes']:,} bytes / {second_run['deleted_files']:,} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

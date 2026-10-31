"""Reclaim storage held by abandoned V4.1 scratch and regenerable packages.

Two classes of bytes are removed, both of which are reproducible rather than
authoritative:

* **Abandoned page-backed scratch.** ``validate_deepseek_v41_production_model``
  backs the multi-gigabyte embedding, head and packed block weights with
  ``np.memmap`` files under the OS temp directory. On Windows the mapping keeps
  its file handle (and therefore the name) alive until every view is dropped, so
  an interrupted run leaves the directory behind. Those directories are pure
  scratch: regenerating them costs one oracle pass and no frozen evidence.

* **Regenerable generated packages.** Wide/structure/production/LoRA shard
  packages under ``validation/fixtures/`` are rebuilt from their recorded seeds
  and CPU oracles. Their frozen expectations (``*_model.json``), the retained
  scalable release checkpoint and every shader artifact stay.

Every deletion is an individual file under an allowed root; directories are only
removed once empty. The ledger records the byte count per root so the reclaim is
auditable against the previous ``checkpoint-cleanup.json`` pass.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "hierarchos-vulkan" / "validation" / "fixtures"
TEMP_ROOT = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
LEDGER = (
    REPO_ROOT
    / "hierarchos-vulkan"
    / "validation"
    / "release_parity"
    / "deepseek_v41"
    / "scratch-cleanup.json"
)

# Generated packages to remove: rebuilt from recorded seeds, and reached only by
# `#[ignore]`d gates, so no default test depends on them.
#
# Deliberately kept:
#   * `deepseek_v41_joint_4141111` - the retained scalable release checkpoint that
#     `qualification.json`, `qualify_deepseek_v41_scalable.py` and the Intel
#     handoff all point at.
#   * `deepseek_v41_long_context_4141112` - small, and referenced by the recorded
#     fixture hashes.
#   * `deepseek_v41_structure_4141101/_4141102` and `deepseek_v41_wide_4141071/
#     _4141072/_4141073` - **non-ignored** tests load these. Removing them fails
#     six `deepseek_v41_*_strict_all_logits` structure tests and
#     `deepseek_v41_model_wider_yarn_bf16_strict_all_logits`. Regenerate with
#     `validate_deepseek_v41_structure_model.py --write-checkpoints` and
#     `validate_deepseek_v41_wide_model.py --write-fixture` if they are lost.
FIXTURE_PACKAGES = [
    "deepseek_v41_production_4141090",
    "deepseek_v41_lora_4141095",
    "deepseek_v41_schedule_width_4141211",
]

# Gitignored PEFT scratch from earlier bank/validation passes.
SCRATCH_DIRS = [
    REPO_ROOT / "hierarchos-vulkan" / ".peft-bank-work",
    REPO_ROOT / "hierarchos-vulkan" / ".peft-validation-work",
]

# `verify_peft_lora_strict.py` used to archive the previous oracle fixture into a
# `.prior-<ns>` sibling on every run, with nothing pruning the archive. The
# newest prior per family is kept as continuation state; older ones are stale.
PEFT_FIXTURE_ROOT = REPO_ROOT / "hierarchos-vulkan" / ".peft-oracle-fixtures"

TEMP_PREFIXES = [
    "deepseek-v41-packed-",
    "deepseek-v41-vocabulary-",
    # One fixture directory per `hierarchos_causal_lm_tests::vendor_fixture()` call.
    "hierarchos-causal-lm-vendor-",
]


def directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def delete_individual_files(root: Path, allowed_parent: Path) -> tuple[int, int]:
    """Delete every file under ``root``, never leaving ``allowed_parent``.

    Returns ``(files_removed, bytes_removed)``. Each candidate is resolved and
    checked against the allowed parent before it is touched, and empty
    directories are pruned bottom-up afterwards.
    """
    resolved_parent = allowed_parent.resolve()
    files = 0
    total = 0
    for entry in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not entry.is_file():
            continue
        try:
            resolved = entry.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(resolved_parent):
            raise RuntimeError(f"refusing to delete outside {resolved_parent}: {resolved}")
        try:
            size = entry.stat().st_size
            entry.unlink()
        except OSError as error:
            print(f"  skipped {entry}: {error}", file=sys.stderr)
            continue
        files += 1
        total += size
    for entry in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if entry.is_dir():
            try:
                entry.rmdir()
            except OSError:
                pass
    return files, total


def main() -> int:
    before = shutil.disk_usage(str(REPO_ROOT)).free
    ledger: dict[str, object] = {
        "method": "individual generated files under an allowed root; directories only once empty",
        "freed_before_probe": before,
        "roots": [],
    }
    removed_files = 0
    removed_bytes = 0

    targets: list[tuple[Path, Path]] = []
    for name in FIXTURE_PACKAGES:
        candidate = FIXTURES / name
        if candidate.is_dir():
            targets.append((candidate, FIXTURES))
    for candidate in SCRATCH_DIRS:
        if candidate.is_dir():
            targets.append((candidate, candidate.parent))
    for prefix in TEMP_PREFIXES:
        for candidate in sorted(TEMP_ROOT.glob(f"{prefix}*")):
            if candidate.is_dir():
                targets.append((candidate, candidate.parent))
    if PEFT_FIXTURE_ROOT.is_dir():
        priors = sorted(
            entry
            for entry in PEFT_FIXTURE_ROOT.iterdir()
            if entry.is_dir() and ".prior-" in entry.name
        )
        newest: dict[str, tuple[int, str]] = {}
        for candidate in priors:
            family, _, stamp = candidate.name.partition(".prior-")
            try:
                order = int(stamp)
            except ValueError:
                order = -1
            if family not in newest or order > newest[family][0]:
                newest[family] = (order, candidate.name)
        for candidate in priors:
            family = candidate.name.partition(".prior-")[0]
            if newest[family][1] != candidate.name:
                targets.append((candidate, PEFT_FIXTURE_ROOT))

    for target, allowed_parent in targets:
        measured = directory_bytes(target)
        files, total = delete_individual_files(target, allowed_parent)
        try:
            target.rmdir()
        except OSError:
            pass
        removed_files += files
        removed_bytes += total
        ledger["roots"].append(
            {
                "path": str(target),
                "allowed_root": str(allowed_parent),
                "measured_bytes": measured,
                "files_removed": files,
                "bytes_removed": total,
                "removed": not target.exists(),
            }
        )
        print(f"{total:>14,} bytes  {files:>5} files  {target}")

    after = shutil.disk_usage(str(REPO_ROOT)).free
    # Keep every run: a reclaim ledger that only records the most recent pass
    # cannot account for the storage the earlier passes already recovered.
    history: list[dict[str, object]] = []
    if LEDGER.is_file():
        try:
            previous = json.loads(LEDGER.read_text(encoding="utf-8"))
            history = list(previous.get("runs", []))
            if not history and previous.get("roots"):
                history = [
                    {
                        "finished_free_bytes": previous.get("free_after"),
                        "started_free_bytes": previous.get("free_before"),
                        "deleted_bytes": previous.get("deleted_bytes"),
                        "deleted_files": previous.get("deleted_files"),
                        "roots": previous.get("roots", []),
                    }
                ]
        except (OSError, ValueError):
            history = []
    ledger["deleted_bytes"] = removed_bytes
    ledger["deleted_files"] = removed_files
    ledger["free_before"] = before
    ledger["free_after"] = after
    ledger["reclaimed_reported_bytes"] = after - before
    ledger["retained_scalable_checkpoint"] = str(FIXTURES / "deepseek_v41_joint_4141111")
    ledger["retained_frozen_expectations"] = sorted(
        entry.name for entry in FIXTURES.glob("*.json")
    )
    ledger["runs"] = history + [
        {
            "started_free_bytes": before,
            "finished_free_bytes": after,
            "deleted_bytes": removed_bytes,
            "deleted_files": removed_files,
            "roots": ledger["roots"],
        }
    ]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"removed {removed_files} files, {removed_bytes:,} bytes")
    print(f"free before {before:,}  after {after:,}  delta {after - before:,}")
    print(f"ledger {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

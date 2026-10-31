#!/usr/bin/env python3
"""Check the committed training-stack receipt against the ladder it records.

`verify_training_stack.py --receipt` writes one row per step: the step name, its
status, and a detail line. That file is the only durable record of a green
ladder, so it is worth exactly as much as the check that reads it. This guard is
that check, on the CPU and without a GPU:

* the receipt holds exactly the ladder's steps, in ladder order -- a run that
  silently drops a step, renames one, or gains one is a different ladder, and a
  receipt that no longer describes `verify_training_stack.py` must not pass;
* every step is `PASS`. A `FAIL` or `BLOCKED` row is a real outcome the ladder
  reported, and committing it as the checked artifact would turn the receipt
  into a list of things that used to work;
* every coverage count in the detail line meets the floor recorded here. The
  floors are the counts the ladder verifies -- golden invocations, matched
  shader variants, comparison modules, training-state runs and exact tensor
  digests -- so a run that exercises fewer cases fails even when every remaining
  case passes. A count that grows is fine: the ladder measured more than the
  floor. `--record` rewrites the committed receipt and prints the counts it
  adopted, which is the deliberate act that accepts a new baseline; the floors
  stay this file's minima, so a baseline that rose is accepted here and written
  into `COVERAGE` in the same diff.

The floors live here rather than in the receipt so that editing the receipt by
hand cannot lower them: changing a count in the committed file makes this check
fail until the same change is made to the spec, in one reviewable diff.

Usage:
    python hierarchos-vulkan/validation/check_training_stack_receipt.py
    python hierarchos-vulkan/validation/check_training_stack_receipt.py --receipt path.json
    python hierarchos-vulkan/validation/check_training_stack_receipt.py \\
        --compare target/debug_parity.json
    python hierarchos-vulkan/validation/check_training_stack_receipt.py --record target/debug_parity.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "validation" / "training_stack_receipt.json"

# The ladder's steps, in the order `verify_training_stack.py` runs them. Names
# come from the `Step` objects the script builds, not from this file, so the two
# are compared literally: a renamed step fails here rather than drifting.
LADDER_STEPS = (
    "cargo build (ladder binaries)",
    "shader toolchain",
    "binding-access audit",
    "shader provenance",
    "push-constant layouts",
    "oracle parity",
    "golden dense (matmul)",
    "golden reduction",
    "golden attention",
    "training-step probe",
    "topology decisions",
    "training-state parity",
)

# A count the ladder reports in its detail line and the floor it must meet. Each
# pattern captures one integer; the floor is the least a green run may report.
COVERAGE = {
    "binding-access audit": (
        (r"out of (\d+) modules", 471),
    ),
    "shader provenance": (
        (r"(\d+) variant\(s\) match", 464),
    ),
    "golden dense (matmul)": (
        (r"(\d+) invocation\(s\) exercised", 144),
    ),
    "golden reduction": (
        (r"(\d+) invocation\(s\) exercised", 183),
    ),
    "golden attention": (
        (r"(\d+) invocation\(s\) exercised", 79),
    ),
    "training-step probe": (
        (r"(\d+) dispatches", 6512),
    ),
    "topology decisions": (
        (r"(\d+) measured decision\(s\)", 4),
    ),
    "training-state parity": (
        (r"(\d+) suite\(s\)", 2),
        (r"(\d+) run\(s\)", 29),
        (r"(\d+) exact tensor digests", 32162),
    ),
}

# The ladder's device-aware steps name the adapter they measured, and a receipt
# re-recorded on another machine is a different result even when every count
# happens to match: the committed artifact describes one verified device. The
# toolchain line names absolute install paths instead, so it is not compared.
RECORDED_ADAPTER = "Intel(R) HD Graphics 520"
ADAPTER_STEPS = {
    "training-step probe": r"loss [0-9.]+ on (.+)$",
    "topology decisions": r"match the record on '(.+)'$",
}

# Literal results a green run reports. A changed value is a different outcome,
# not a formatting difference, so it fails even when every count still matches.
RESULT_VALUES = {
    "training-step probe": ((r"loss ([0-9.]+) on", "5.5432177"),),
}


class Problem(Exception):
    """A check failed; the message is the report line."""


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise Problem(f"missing {path}; write it with verify_training_stack.py --receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise Problem(f"{path} is not valid JSON: {err}") from err
    if not isinstance(receipt, list) or not receipt:
        raise Problem(f"{path} holds {type(receipt).__name__}, not a list of steps")
    return receipt


def counts(step: str, detail: str) -> dict[str, int]:
    """The coverage counts this step's detail line reports, keyed by pattern."""
    found = {}
    for pattern, _ in COVERAGE.get(step, ()):
        match = re.search(pattern, detail)
        found[pattern] = int(match.group(1)) if match else -1
    return found


def check_structure(receipt: list[dict]) -> None:
    """Exactly the ladder's steps, once each, in ladder order."""
    names = [row.get("step") for row in receipt]
    if len(names) != len(LADDER_STEPS):
        raise Problem(
            f"the receipt records {len(names)} step(s); the ladder runs {len(LADDER_STEPS)}. "
            f"Recorded: {', '.join(str(name) for name in names)}"
        )
    if tuple(names) != LADDER_STEPS:
        missing = [name for name in LADDER_STEPS if name not in names]
        extra = [name for name in names if name not in LADDER_STEPS]
        parts = []
        if missing:
            parts.append("missing " + ", ".join(repr(name) for name in missing))
        if extra:
            parts.append("unexpected " + ", ".join(repr(name) for name in extra))
        if not parts:
            parts.append("the order differs from the ladder")
        raise Problem("the receipt does not describe the ladder: " + "; ".join(parts))


def check_rows(receipt: list[dict]) -> None:
    for row in receipt:
        if row.get("status") != "PASS":
            raise Problem(
                f"{row.get('step')!r} recorded {row.get('status')!r}: {row.get('detail', '')}"
            )
        if not isinstance(row.get("detail"), str) or not row["detail"].strip():
            raise Problem(f"{row.get('step')!r} carries no detail line")


def check_adapter(receipt: list[dict]) -> None:
    """Every device-aware step names the same adapter, and it is the recorded one."""
    named = {}
    for row in receipt:
        pattern = ADAPTER_STEPS.get(row["step"])
        if pattern is None:
            continue
        match = re.search(pattern, row["detail"])
        if not match:
            raise Problem(
                f"{row['step']!r} no longer names the adapter it measured "
                f"(pattern {pattern!r}); the detail line changed shape"
            )
        named[row["step"]] = match.group(1).strip()
    if not named:
        raise Problem("the receipt names no adapter; the ladder reports one on every run")
    values = sorted(set(named.values()))
    if len(values) > 1:
        raise Problem(
            "the device-aware steps disagree about the adapter: "
            + ", ".join(f"{step} {name!r}" for step, name in sorted(named.items()))
        )
    if values[0] != RECORDED_ADAPTER:
        raise Problem(
            f"the receipt was recorded on {values[0]!r}, not the verified adapter "
            f"{RECORDED_ADAPTER!r}; change RECORDED_ADAPTER here in the same diff if "
            "the gate moved machines, then re-record"
        )


def check_results(receipt: list[dict]) -> None:
    """The literal results the ladder reported are the ones the receipt records."""
    for row in receipt:
        for pattern, expected in RESULT_VALUES.get(row["step"], ()):
            match = re.search(pattern, row["detail"])
            if not match:
                raise Problem(
                    f"{row['step']!r} no longer reports the result this guard reads "
                    f"(pattern {pattern!r}); the detail line changed shape"
                )
            if match.group(1) != expected:
                raise Problem(
                    f"{row['step']!r} reports {match.group(1)!r}, the receipt records "
                    f"{expected!r}: the run produced a different result"
                )


def check_coverage(receipt: list[dict], floors: dict[str, tuple[tuple[str, int], ...]] = COVERAGE) -> None:
    for row in receipt:
        step = row["step"]
        if step not in floors:
            continue
        reported = counts(step, row["detail"])
        for pattern, floor in floors[step]:
            value = reported[pattern]
            if value < 0:
                raise Problem(
                    f"{step!r} no longer reports the count this guard reads "
                    f"(pattern {pattern!r}); the detail line changed shape"
                )
            if value < floor:
                raise Problem(
                    f"{step!r} reports {value} for {pattern!r}, below the recorded floor "
                    f"{floor}. The ladder is exercising fewer cases than the receipt "
                    "documents; re-record deliberately with --record if that is intended"
                )


def committed_floors(receipt: list[dict]) -> dict[str, tuple[tuple[str, int], ...]]:
    """The counts in one receipt, as floors a later run must still meet."""
    floors = {}
    for row in receipt:
        step = row["step"]
        if step not in COVERAGE:
            continue
        reported = counts(step, row["detail"])
        floors[step] = tuple(
            (pattern, reported[pattern]) for pattern, _ in COVERAGE[step]
        )
    return floors


def check_compare(fresh: list[dict], committed: list[dict]) -> list[str]:
    """A fresh ladder run against the committed receipt.

    Every committed step must still be there, still pass, and still report at
    least the coverage it reported before. Counts that grew are reported so the
    receipt can be re-recorded on purpose.
    """
    check_structure(fresh)
    check_rows(fresh)
    check_adapter(fresh)
    check_results(fresh)
    check_coverage(fresh)
    check_coverage(fresh, committed_floors(committed))
    report = []
    for old, new in zip(committed, fresh):
        for pattern, floor in committed_floors([old]).get(old["step"], ()):
            value = counts(new["step"], new["detail"])[pattern]
            if value > floor:
                report.append(
                    f"{new['step']!r} covers {value} (receipt records {floor}); "
                    "re-record to accept the larger baseline"
                )
    return report


def record(path: Path, fresh: list[dict]) -> None:
    """Adopt a fresh run as the committed receipt, after the same checks."""
    check_structure(fresh)
    check_rows(fresh)
    check_adapter(fresh)
    check_results(fresh)
    check_coverage(fresh)
    path.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    floors = committed_floors(fresh)
    print(f"recorded {len(fresh)} step(s) in {path}")
    for step in sorted(floors):
        counts_text = ", ".join(f"{value}" for _, value in floors[step])
        print(f"  {step}: {counts_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument(
        "--compare",
        type=Path,
        help="a fresh ladder receipt to compare against the committed one",
    )
    parser.add_argument(
        "--record",
        type=Path,
        metavar="FRESH",
        help="adopt a fresh ladder receipt as the committed one",
    )
    args = parser.parse_args(argv)
    try:
        committed = load(args.receipt)
        if args.record is not None:
            record(args.receipt, load(args.record))
            return 0
        check_structure(committed)
        check_rows(committed)
        check_adapter(committed)
        check_results(committed)
        check_coverage(committed)
        if args.compare is not None:
            report = check_compare(load(args.compare), committed)
            for line in report:
                print(f"note: {line}")
        print(
            f"PASS: {len(committed)} step(s) PASS in {args.receipt.name} on "
            f"{RECORDED_ADAPTER} and the coverage floors hold"
        )
        covered = sum(len(COVERAGE.get(row["step"], ())) for row in committed)
        print(f"PASS: {covered} coverage count(s) at or above their floors")
    except Problem as err:
        print(f"FAIL: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

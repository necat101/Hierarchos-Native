"""Check complete training trajectories against pre-optimization GPU results.

Goldens were obtained from the pre-optimization shaders and runtime, retaining
the earlier CE/objective work. They cover every parameter, both Adam moments,
per-slot step counters, losses, recurrent states and initial-state adjoints after
each update. Unlike a loss-only check, this catches drift before it affects loss.
No command in this verifier records or updates goldens.

There is one suite per training numerical policy, because the policy selects a
different kernel family: the FP32 suite runs the FP32 LN1/low-rank producer and
the FP16-storage suite runs the packed-FP16 one, and their trajectories differ
(FP16 storage rounds the operands), so each has its own device-keyed golden and
its own arms.

Inside a suite the whole trajectory set runs once per arm: the shipped (measured)
choice first, then every LN1/low-rank topology variant forced through
`HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY`. Every variant is supposed to be
bit-identical to the kernel the trajectories were recorded with, so an arm that
disagrees fails exactly like a default run would.

Forcing is not proof by itself. A variant this device cannot run is rejected by the
producer, which then runs the default kernel -- a run that passes and proves
nothing. So each arm's probe stream is read for the request the producer
acknowledged, and an arm is only credited where the value it asked for is the value
that ran. An arm whose variant this device rejects in every case is a note: the
device cannot exercise it, and the coverage line says so. An arm that ran nowhere
and was never rejected is a failure, because then the fused producer never
dispatched at all and nothing was verified.

What the arms exercise is itself checked in, in
`validation/training_state_coverage.json`: for every (suite, device) it records the
golden cases that were measured, how much each one compares, and which arm ran in
which case. A run compares its own coverage against that record and fails when it
shrinks -- an arm or a case that stops being exercised, or a golden case that
compares fewer steps or tensors than it did -- so a variant that quietly stops
being tested is a regression rather than a smaller fraction in a log line. A device
with no recorded coverage fails too, and `--record-coverage` merges this machine's
coverage in, refusing to shrink an existing entry unless `--allow-coverage-shrink`
says the loss is intended. Like the goldens, that file holds other machines'
evidence: only the entries this run measured are replaced. `--bootstrap-coverage` is
the deliberate first run on a new device: it creates only the entries this machine is
missing, prints exactly what it added, and leaves every existing entry as recorded.

The record's audit, merge, shrink refusal and bootstrap are not this driver's code.
They live in `src/coverage_record.rs`, the same implementation the kernel golden
harnesses use, and this driver asks it through the `coverage_record` tool that ships
next to the probe (`--coverage-tool` names another). Keeping one implementation is
what stops one record's shrink rule from drifting from the other's.

The record also says what an arm is *not* worth running. Two verdicts say the variant
is not there on this device: the producer rejected the request (`unavailable`), or the
fused producer never dispatched at that shape at all (`not_dispatched`) -- the FP32
producer is only enabled up to rank 64, so this model's width-256 case is the second
kind. Forcing a variant in such a case only measures the fallback kernel, so a forced
arm runs every case the record has it verified in, every case the record says nothing
about, and one **canary** per arm, and the remaining absent cases are carried from the
record as `asserted` instead of re-measured. The canary is the widest of those cases
-- the shape nearest the boundary where the variant would appear if this device
changed -- and it replays a single update, because the verdict is settled by the first
dispatch and there is no trajectory to compare in a case the variant never ran in. So
the claim stays device evidence rather than an assumption: every run re-checks it on
this machine, and when the canary disagrees the run measures that arm's carried cases
in full instead.

Every arm of a case also runs against one shared model package. The probe
initializes a package when it is missing and only reads it afterwards, so the
initial weights are byte-identical for every arm and paying to initialize one
directory per (arm, case) bought nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
from typing import NamedTuple

CRATE = Path(__file__).resolve().parents[1]
GOLDEN_DIR = CRATE / "validation" / "kernel_golden"
COVERAGE = CRATE / "validation" / "training_state_coverage.json"
FORCED_TOPOLOGY_ENV = "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY"
AUTOTUNE_LOG_ENV = "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG"
PRECISION_ENV = "HIERARCHOS_VULKAN_TRAINING_PRECISION"
# The accepted request and the rejected one carry the value the caller asked for, not
# just the variant it maps to, so an arm can be attributed without a second copy of
# the value-to-variant table living here. Both families print these two lines, which
# is what lets one arm list cover the FP32 and packed-FP16 producers.
FORCED_LINE = re.compile(r"forced topology=(\S+) requested=(\S+)")
UNAVAILABLE_LINE = re.compile(r"topology=(\S+) requested=(\S+) is unavailable")
MEASURED_ARM = "measured"


class Suite(NamedTuple):
    """One training numerical policy's trajectory golden and the arms it is checked with.

    The policy itself comes from the golden's own `env` block rather than from here, so
    the environment a suite was recorded under cannot drift from the one it is replayed
    under. The arms are policy here, next to the harness that runs them.
    """

    name: str
    golden: Path
    arms: tuple[str, ...]


SUITES = (
    Suite(
        "fp32",
        GOLDEN_DIR / "training_state.json",
        ("one-row", "two-rows", "column-slices"),
    ),
    Suite(
        "fp16-storage-fp32-compute",
        GOLDEN_DIR / "training_state_fp16.json",
        (
            "one-row",
            "one-row-cached",
            "one-row-rank128",
            "two-rows",
            "two-rows-wide",
            "four-rows",
        ),
    ),
)

# The probe's own default, used when a golden case leaves `--hidden` out.
PROBE_DEFAULT_HIDDEN = 128

# How many updates a canary replays. A canary re-checks whether a variant exists at a
# shape on this device, which is settled by the first dispatch, so a full trajectory
# there buys nothing: the record already has the variant absent in that case, which is
# why there is no trajectory to compare in it.
CANARY_STEPS = 1


def compare_snapshots(expected, actual):
    errors = []
    if len(expected) != len(actual):
        errors.append(f"step count {len(actual)} != {len(expected)}")
    for index, (reference, observed) in enumerate(zip(expected, actual)):
        for field in ("step", "optimizer_step", "optimizer_slot_steps"):
            if reference.get(field) != observed.get(field):
                errors.append(f"step {index}: {field} differs")
        before = reference.get("tensors", {})
        after = observed.get("tensors", {})
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                errors.append(f"step {index}: {name} differs")
    return errors


def arm_evidence(stderr: str, arm: str) -> tuple[bool, bool]:
    """Did this case run the forced variant, and did the producer reject the request?

    Returns `(exercised, unavailable)`. Both come from the producer's own log lines:
    the one it prints when it accepts a forced topology, and the one it prints when
    it cannot offer the requested variant on this device and falls back.
    """
    exercised = any(match.group(2) == arm for match in FORCED_LINE.finditer(stderr))
    unavailable = any(match.group(2) == arm for match in UNAVAILABLE_LINE.finditer(stderr))
    return exercised, unavailable


def forced_arm_report(rows, topologies):
    """Per-arm coverage: where the variant ran, and where this device had no place for it.

    An arm that ran the variant in at least one case is verified there, and the cases it
    fell back in are reported beside it because they say nothing about the variant either
    way. An arm the fused producer never dispatched for is a note rather than a failure:
    a variant that only exists at a shape this model does not reach is a fact about the
    shape, and it is the checked-in coverage record that fails when a variant which *was*
    verified on this device stops running, which is the regression worth catching.
    """
    notes = []
    problems = []
    for arm in topologies:
        arm_rows = [row for row in rows if row["arm"] == arm]
        exercised = sorted({row["case"] for row in arm_rows if row["forced"]})
        rejected = [
            row["case"] for row in arm_rows if row["unavailable"] and not row["forced"]
        ]
        absent = [
            row["case"] for row in arm_rows if not row["dispatched"] and not row["forced"]
        ]
        if exercised:
            notes.append(
                f"{arm}: the forced variant ran in {len(exercised)}/{len(arm_rows)} case(s) "
                f"({', '.join(exercised)})"
            )
            if rejected:
                notes.append(
                    f"{arm}: this device reported the variant unavailable in "
                    f"{len(rejected)} case(s), which kept their default kernel "
                    f"({', '.join(rejected)})"
                )
        elif rejected:
            notes.append(
                f"{arm}: this device reports the variant unavailable in every measured "
                f"case ({', '.join(rejected)}), so the arm verifies nothing here"
            )
        if absent:
            notes.append(
                f"{arm}: the fused producer did not dispatch the request in "
                f"{len(absent)} case(s) ({', '.join(absent)}), so forcing the variant "
                "measures nothing there"
            )
        if not exercised and not rejected and not absent:
            problems.append(
                f"{arm}: this run has no verdict for the arm at all, so the fused "
                "producer did not dispatch and nothing was verified; pass --topologies "
                "for the arms this device can run"
            )
    return notes, problems


def load_golden(path):
    recorded = json.loads(Path(path).read_text(encoding="utf-8"))
    if recorded.get("schema") != 1 or not recorded.get("cases"):
        raise RuntimeError(f"{path}: unsupported or empty training-state golden file")
    return recorded


def fixture_dir(temp, suite, case):
    """The one model package every arm of a golden case runs against.

    The probe initializes a package when `--model-dir` does not already hold one and
    only ever reads it afterwards -- a training run advances the in-memory graph, not
    the files -- so every arm of a case loads the same byte-identical initial weights
    from a single directory. One package per (arm, case) re-created those same weights
    for every arm instead, which is a measurable part of a short run.
    """
    return temp / f"fixture_{suite.name}_{case['name']}"


def run_case(probe, suite, recorded, case, temp, label, arm, steps=None):
    """Run one case under one arm and compare it against this device's trajectory.

    `steps` shortens the run, which is what a canary needs: it re-checks whether this
    device offers the arm's variant at that shape, and that is a property of the shape
    and the override, not of how many updates are replayed. The comparison then covers
    the steps that ran, so a shortened canary verifies less than a full run and is only
    ever used where the record already says the variant never ran.
    """
    snapshot = temp / f"state_{suite.name}_{case['name']}_{label}.json"
    environment = dict(os.environ)
    environment.update(recorded.get("env", {}))
    environment[AUTOTUNE_LOG_ENV] = "1"
    if arm is None:
        environment.pop(FORCED_TOPOLOGY_ENV, None)
    else:
        environment[FORCED_TOPOLOGY_ENV] = arm
    args = list(case["args"])
    if steps is not None:
        args[args.index("--steps") + 1] = str(steps)
    command = [str(probe), "--model-dir", str(fixture_dir(temp, suite, case)),
               *args, "--state-output", str(snapshot)]
    result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True, env=environment)
    if result.returncode:
        raise RuntimeError(
            f"{suite.name} {label} {case['name']} probe exit {result.returncode}: "
            f"{result.stderr[-3000:]}"
        )
    output = json.loads(result.stdout[result.stdout.index("{"):])
    device = output["device"]
    if device not in case["devices"]:
        raise RuntimeError(
            f"{suite.name} {case['name']}: no trajectory recorded for {device}"
        )
    exercised, unavailable = (False, False) if arm is None else arm_evidence(result.stderr, arm)
    expected = case["devices"][device]
    actual = json.loads(snapshot.read_text(encoding="utf-8"))
    compared = expected[: len(actual)]
    errors = compare_snapshots(compared, actual)
    return {
        "suite": suite.name,
        "case": case["name"],
        "arm": label,
        "device": device,
        "steps": len(actual),
        # What this row actually compared, so a shortened canary cannot inflate the run's
        # digest count. The coverage record's per-case counts come from the measured arm,
        # which always replays every case in full.
        "tensor_comparisons": sum(len(step["tensors"]) for step in compared),
        "forced": exercised,
        "unavailable": unavailable,
        # A requested variant can be missing from a case for two different reasons, and
        # the record keeps them apart: this device rejected the request (`unavailable`),
        # or the fused producer never saw it at that shape at all. The second is not a
        # rejection -- there is no fallback to compare -- so the arm is not evidence
        # there either way, but a later run may carry it just like a rejection.
        "dispatched": exercised or unavailable,
        "shortened": steps is not None,
        "pass": not errors,
        "errors": errors,
    }


def case_flag(case, flag, default):
    """One numeric probe flag of a golden case, or the probe's default for it."""
    args = case["args"]
    return int(args[args.index(flag) + 1]) if flag in args else default


def recorded_arm(entry, arm):
    """This device's recorded entry for one arm, or `None` when it has none."""
    if entry is None:
        return None
    return entry["arms"].get(arm)


def verified_cases(entry, arm):
    """Cases this device has already verified the arm's variant in."""
    recorded = recorded_arm(entry, arm)
    return set(recorded.get("exercised", ())) if recorded else set()


def claimed_absent(entry, arm):
    """Cases this device's record says the arm's variant is not running in.

    Every half counts: the rejections a run confirmed (`unavailable`), the shapes the
    fused producer never dispatched in (`not_dispatched`), and the verdicts a previous
    run carried (`asserted`). A case in any of them is a case where forcing this arm is
    not a measurement of the variant on this device.
    """
    recorded = recorded_arm(entry, arm)
    if not recorded:
        return set()
    return (
        set(recorded.get("unavailable", ()))
        | set(recorded.get("not_dispatched", ()))
        | set(recorded.get("asserted", ()))
    )


def canary_case(cases, claimed):
    """The name of the case one arm's recorded unavailability is re-checked in.

    The widest geometry among the cases the record claims the variant unavailable in,
    then the one measuring the most steps. Availability is a property of the shape the
    producer can dispatch, so the case nearest the boundary is where a device that
    changed would show it first, and the deepest case is the strongest evidence that it
    did not. `cases` is in golden order, so that breaks remaining ties.
    """
    candidates = [case for case in cases if case["name"] in claimed]
    if not candidates:
        return None
    order = {case["name"]: index for index, case in enumerate(cases)}
    chosen = max(
        candidates,
        key=lambda case: (
            case_flag(case, "--hidden", PROBE_DEFAULT_HIDDEN),
            case_flag(case, "--steps", 0),
            -order[case["name"]],
        ),
    )
    return chosen["name"]


def arm_plan(cases, arm, entry):
    """Which cases one forced arm runs, and which the record says it need not.

    Returns `(must_run, canary, skipped)`. A case is run when the record has the arm
    verified there -- that verification has to keep holding -- or when the record says
    nothing about the arm in it, because there is no verdict to lean on. The cases the
    record calls unavailable are skipped except for the canary, which re-checks that
    verdict on this device rather than trusting it. A device with no entry runs every
    case, so the first run on a machine measures everything it can.
    """
    verified = verified_cases(entry, arm)
    claimed = claimed_absent(entry, arm)
    must_run = [
        case["name"] for case in cases
        if case["name"] in verified or case["name"] not in claimed
    ]
    skipped = [
        case["name"] for case in cases
        if case["name"] in claimed and case["name"] not in verified
    ]
    return must_run, canary_case(cases, set(skipped)), skipped


def run_suite(probe, suite, recorded, arms, covered=None):
    """Run the shipped choice over every case, then each arm over what it must.

    The shipped choice runs first and over the whole golden: it is the reference every
    arm is compared against, and it is what names the device whose record decides which
    cases the forced arms have to pay for. Returns `(rows, notes, asserted)`, where
    `asserted` maps each arm to the claimed-unavailable cases this run carried.
    """
    scratch = CRATE / "target" / "verify_training_stack"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    notes = []
    asserted = {}
    with tempfile.TemporaryDirectory(prefix=f"state-parity-{suite.name}-", dir=scratch) as temp:
        temp = Path(temp)
        cases = recorded["cases"]
        by_name = {case["name"]: case for case in cases}
        for case in cases:
            rows.append(run_case(probe, suite, recorded, case, temp, MEASURED_ARM, None))
        device = rows[0]["device"]
        entry = next(
            (
                entry for entry in (covered or ())
                if entry.get("suite") == suite.name and entry.get("device") == device
            ),
            None,
        )
        for arm in arms:
            must_run, canary, skipped = arm_plan(cases, arm, entry)
            if skipped:
                notes.append(
                    f"{arm}: {len(skipped)} case(s) this device's record has the variant "
                    f"absent in carried, re-checked in {canary} ({', '.join(skipped)})"
                )
            # The canary runs first and short: it is there to re-check a dispatch verdict,
            # so a record whose claim no longer holds does not cost this suite a full run
            # that would report a stale verdict as if it were a verification.
            ordered = ([canary] if canary else []) + [name for name in by_name if name in must_run]
            for name in ordered:
                rows.append(
                    run_case(
                        probe, suite, recorded, by_name[name], temp, arm, arm,
                        steps=CANARY_STEPS if name == canary and skipped else None,
                    )
                )
                if name == canary and skipped and rows[-1]["forced"]:
                    notes.append(
                        f"{arm}: the record has the variant absent in {', '.join(skipped)} "
                        f"but it ran in {canary}, so this run measures them again instead "
                        f"of carrying the claim"
                    )
                    # The contradiction makes the record's claim stale, and the short
                    # canary has no trajectory worth keeping, so every case is measured in
                    # full -- the canary included, which is what the record was missing.
                    rows.pop()
                    for extra in [canary, *skipped]:
                        rows.append(
                            run_case(probe, suite, recorded, by_name[extra], temp, arm, arm)
                        )
                    skipped = []
            # The canary was measured -- it is where this run confirmed or refused the
            # record's claim -- so only the cases it stood in for are carried.
            asserted[arm] = [name for name in skipped if name != canary]
    return rows, notes, asserted


def coverage_entry(suite, device, rows, cases, asserted=None):
    """What this (suite, device) run exercised: its cases and where each arm ran.

    `asserted` maps an arm to the cases this run did not measure because this device's
    record already reports the variant unavailable there. They are written apart from
    the `unavailable` cases the run confirmed itself, so the record says which verdicts
    are this run's evidence and which are the entry's own claim being carried forward
    by a canary re-check.
    """
    entry = {"suite": suite.name, "device": device, "cases": {}, "arms": {}}
    for name in cases:
        measured = next(
            row for row in rows if row["arm"] == MEASURED_ARM and row["case"] == name
        )
        entry["cases"][name] = {
            "steps": measured["steps"],
            "tensor_comparisons": measured["tensor_comparisons"],
        }
    for arm in suite.arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        entry["arms"][arm] = {
            "exercised": [row["case"] for row in arm_rows if row["forced"]],
            "unavailable": [
                row["case"] for row in arm_rows if row["unavailable"] and not row["forced"]
            ],
            "not_dispatched": [
                row["case"] for row in arm_rows if not row["dispatched"] and not row["forced"]
            ],
            "asserted": list((asserted or {}).get(arm, ())),
        }
    return entry


def load_coverage(path):
    """Read the coverage entries this run plans against.

    Only planning needs this; the file's contract -- its schema, its key uniqueness, its
    entries' fields, and everything the audit and the merge may and may not do -- belongs
    to `src/coverage_record.rs` and is enforced by the coverage record tool. A record this
    reader cannot parse is returned as no entries, which makes the run measure every arm
    and every case, and the tool then reports the real problem in the same pass.
    """
    path = Path(path)
    if not path.is_file():
        return [], [f"{path} is missing; create it with --record-coverage on a device"]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], [f"{path}: {error}"]
    entries = document.get("coverage") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return [], [f"{path}: coverage must be a list"]
    return [entry for entry in entries if isinstance(entry, dict)], []


def coverage_tool(probe):
    """The coverage record tool that ships next to the probe binary.

    Both coverage records are audited, merged and refused by one implementation, so this
    driver delegates to it instead of keeping a second copy of those rules.
    """
    probe = Path(probe)
    return probe.with_name("coverage_record" + (probe.suffix if os.name == "nt" else ""))


def run_coverage_tool(tool, mode, path, entries, bootstrap=False, allow_shrink=False):
    """Ask the shared record engine to audit or record, and return its response."""
    request = json.dumps(
        {
            "mode": mode,
            "file": str(path),
            "bootstrap": bool(bootstrap),
            "allow_shrink": bool(allow_shrink),
            "entries": list(entries),
        }
    )
    try:
        completed = subprocess.run(
            [str(tool)], input=request, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
    except OSError as error:
        raise SystemExit(
            f"cannot run the coverage record tool {tool}: {error}\n"
            "build it with `cargo build --bin coverage_record`, or name one with "
            "--coverage-tool"
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"coverage record tool {tool} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def absent_cases(recorded_arm_entry):
    """The cases one arm's entry says the variant is not running in on this device.

    Two verdicts, kept apart because they mean different things: the producer rejected
    the request (`unavailable`), or the fused producer never dispatched at that shape at
    all (`not_dispatched`). Either way the arm is not evidence there.
    """
    return set(recorded_arm_entry.get("unavailable", ())) | set(
        recorded_arm_entry.get("not_dispatched", ())
    )


def verify(probe, suites=SUITES, coverage=COVERAGE, record=False, allow_shrink=False,
           enforce_coverage=True, bootstrap=False, tool=None):
    """Run every suite and audit both the trajectories and the coverage record."""
    recorded_coverage, _ = load_coverage(coverage)
    tool = coverage_tool(probe) if tool is None else Path(tool)
    # Suite-scoped problems are kept apart from the coverage and record problems so a
    # report can say which suite failed without blaming the others for it.
    suite_problems = {suite.name: [] for suite in suites}
    problems = []
    notes = []
    rows = []
    entries = []
    devices = set()
    for suite in suites:
        golden = Path(suite.golden)
        if not golden.is_file():
            suite_problems[suite.name].append(f"{golden} is missing; record its trajectories")
            continue
        recorded = load_golden(golden)
        cases = [case["name"] for case in recorded["cases"]]
        suite_rows, gating_notes, asserted = run_suite(
            probe,
            suite,
            recorded,
            suite.arms,
            # Gating runs on the record, so a run that does not audit the record must not
            # lean on it either: `--topologies` names the arms to run by hand.
            recorded_coverage if enforce_coverage else None,
        )
        notes.extend(f"{suite.name}: {note}" for note in gating_notes)
        arm_notes, arm_problems = forced_arm_report(suite_rows, suite.arms)
        notes.extend(f"{suite.name}: {note}" for note in arm_notes)
        suite_problems[suite.name].extend(arm_problems)
        for row in suite_rows:
            if not row["pass"]:
                suite_problems[suite.name].append(
                    f"{row['arm']} {row['case']}: " + "; ".join(row["errors"][:8])
                )
        rows.extend(suite_rows)
        suite_devices = {row["device"] for row in suite_rows}
        if len(suite_devices) != 1:
            suite_problems[suite.name].append(
                f"one run reported several devices: {sorted(suite_devices)}"
            )
            continue
        devices |= suite_devices
        entries.append(
            coverage_entry(suite, suite_devices.pop(), suite_rows, cases, asserted)
        )
    for name, messages in suite_problems.items():
        problems.extend(f"{name}: {message}" for message in messages)

    if enforce_coverage and entries:
        # Both coverage records share one implementation of the audit, the merge, the shrink
        # refusal and the bootstrap, in the crate; this driver asks it rather than keeping a
        # second copy of those rules in Python.
        response = run_coverage_tool(
            tool,
            "record" if record else "audit",
            coverage,
            entries,
            bootstrap=bootstrap,
            allow_shrink=allow_shrink,
        )
        problems.extend(response["problems"])
        notes.extend(response["notes"])
        if response["written"]:
            notes.append(
                f"recorded coverage for {len(entries)} suite(s) on "
                f"{', '.join(sorted(devices))!r}; the record now holds "
                f"{response['total']} entries ({response['added']} added, "
                f"{response['replaced']} replaced)"
            )
    elif entries and not enforce_coverage:
        notes.append(
            "coverage was not audited because --topologies restricted the arms this run forced"
        )

    return {
        "pass": not problems,
        "suites": [
            {
                "name": suite.name,
                "golden": str(suite.golden),
                "arms": list(suite.arms),
                "device": next(
                    (row["device"] for row in rows if row["suite"] == suite.name), None
                ),
                "cases": [
                    row["case"] for row in rows
                    if row["suite"] == suite.name and row["arm"] == MEASURED_ARM
                ],
            }
            for suite in suites
            if any(row["suite"] == suite.name for row in rows)
        ],
        "cases": rows,
        "coverage": entries,
        "suite_problems": {name: list(messages) for name, messages in suite_problems.items()},
        "notes": notes,
        "problems": problems,
    }


def describe_suite(report, suite):
    """One line per suite: what it compared and where each arm ran.

    An arm's `covered/total` counts the cases this run measured it in; `carried` is how
    many of that arm's claimed-unavailable cases this run took from the record instead,
    each of which the arm's canary re-checked in one case.
    """
    suite_rows = [row for row in report["cases"] if row["suite"] == suite.name]
    measured = [row for row in suite_rows if row["arm"] == MEASURED_ARM]
    steps = sum(row["steps"] for row in measured)
    digests = sum(row["tensor_comparisons"] for row in measured)
    entry = next(
        (entry for entry in report.get("coverage", ()) if entry["suite"] == suite.name), None
    )
    arms = list(suite.arms)
    coverage = []
    for arm in arms:
        arm_rows = [row for row in suite_rows if row["arm"] == arm]
        covered = sum(1 for row in arm_rows if row["forced"])
        carried = len(entry["arms"].get(arm, {}).get("asserted", ())) if entry else 0
        detail = f"{arm} {covered}/{len(arm_rows)}"
        if carried:
            detail += f" +{carried} carried"
        coverage.append(detail)
    detail = (
        f"{len(arms) + 1} arm(s) x {steps} training steps, {digests} exact tensor digests "
        f"each ({MEASURED_ARM} first)"
    )
    if coverage:
        detail += "; forced arms exercised: " + ", ".join(coverage)
    return detail


def select_suites(suites, names):
    if not names:
        return suites
    known = {suite.name for suite in suites}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; available: {sorted(known)}")
    return tuple(suite for suite in suites if suite.name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, default=CRATE / "target" / "debug" /
                        ("training_step_probe.exe" if os.name == "nt" else "training_step_probe"))
    parser.add_argument("--suite", action="append", default=[],
                        help="run only this suite (repeatable; default: every suite)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coverage", type=Path, default=COVERAGE,
                        help="the checked-in per-device coverage record")
    parser.add_argument("--record-coverage", action="store_true",
                        help="merge this device's coverage into the record")
    parser.add_argument("--bootstrap-coverage", action="store_true",
                        help="create only the entries this device is missing, print the diff, "
                             "and leave every existing entry as recorded")
    parser.add_argument("--allow-coverage-shrink", action="store_true",
                        help="record an entry that covers less than the one it replaces")
    parser.add_argument("--coverage-tool", type=Path,
                        help="the coverage record tool (default: coverage_record next to --probe)")
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=None,
        help="topology arms to force and verify (default: every arm of every suite; pass "
             "none to run only the shipped choice, which also skips the coverage audit)",
    )
    args = parser.parse_args()
    if args.record_coverage and args.bootstrap_coverage:
        parser.error("pass one of --record-coverage and --bootstrap-coverage")
    if args.bootstrap_coverage and args.allow_coverage_shrink:
        parser.error("--bootstrap-coverage never replaces an entry, so it cannot shrink one")
    suites = select_suites(SUITES, args.suite)
    restricted = args.topologies is not None
    if restricted:
        suites = tuple(
            suite._replace(arms=tuple(arm for arm in suite.arms if arm in args.topologies))
            for suite in suites
        )
    report = verify(
        args.probe.resolve(),
        suites,
        args.coverage,
        record=args.record_coverage,
        allow_shrink=args.allow_coverage_shrink,
        enforce_coverage=not restricted,
        bootstrap=args.bootstrap_coverage,
        tool=args.coverage_tool,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for note in report["notes"]:
        print(f"note  {note}")
    for problem in report["problems"]:
        print(f"FAIL  {problem}")
    for suite in suites:
        if not any(row["suite"] == suite.name for row in report["cases"]):
            problems = report["suite_problems"].get(suite.name) or ["this suite did not run"]
            print(f"FAIL  {suite.name}: {'; '.join(problems)}")
            continue
        status = "PASS" if not report["suite_problems"].get(suite.name) else "FAIL"
        print(f"{status}  {suite.name}: " + describe_suite(report, suite))
    compared = sum(row["tensor_comparisons"] for row in report["cases"])
    detail = (
        f"{len(suites)} suite(s), {len(report['cases'])} run(s), {compared} exact tensor "
        f"digests compared; coverage audited against {args.coverage}"
        if not restricted
        else f"{len(suites)} suite(s), {len(report['cases'])} run(s); arms restricted, "
             "coverage not audited"
    )
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {detail}")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

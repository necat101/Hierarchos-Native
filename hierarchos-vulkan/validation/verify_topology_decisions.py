"""Replay the measured kernel-topology decisions and audit them against the record.

The LN1/low-rank autotuners choose between kernel variants by timing them on the
executing device. Those choices used to exist only as an env-gated log line, so a
device that silently started selecting a different variant changed the schedule of
every training step without anything on disk saying it had. `topology_decisions.json`
is the checked-in record: one entry per (family, device, geometry) carrying the
medians the choice was made from and the `selected` variant.

This script replays the runs that produce them -- the ladder's training-step fixture
at several geometries, with the reporter's record file pointed at scratch -- and then:

* requires every decision the replay measured to be recorded for this device and to
  still select the same variant. A decision that moves is a failure: the scheduling
  choice is exactly what the record documents;
* treats a sweep the autotuner itself calls contended -- the structural default's own
  samples moving by more than its threshold across the rounds -- as not evidence. On
  this Intel part that is not hypothetical: with a second process on the GPU the
  recorded column-slice win at width 128 failed the margin in every round and the
  replay reported the device selecting one-row. A contended sweep is reported as a
  note, cannot move the record, and is never written to it;
* reports timing drift as a note with the ratio, never as a failure, because a
  median from a shared GPU moves between runs even when the choice does not;
* requires every geometry in the replay to measure something. A geometry is a promise
  that this device decides at that shape, so one that stops measuring fails instead of
  quietly shrinking the record's coverage;
* reports recorded decisions the replay did not exercise (another device, or a
  geometry the replay no longer drives) as notes, so one device can be audited
  without erasing the rest of the record.

The replay drives a shape set rather than one fixture because the choice is per
(width, ranks, batch size) and the variants trade off differently at each: `width 64`
with ranks 8/8/8 is one 64-column slice and the margin keeps the one-row kernel,
while `width 128` at one, two-with-a-tail and sixteen row pairs is where the column
split wins. See `REPLAY_GEOMETRIES`.

`--record` merges this run's decisions into the record, leaving every other device's
entries untouched. Only measured decisions are ever written.

    python validation/verify_topology_decisions.py
    python validation/verify_topology_decisions.py --record
    python validation/verify_topology_decisions.py --probe target/release/training_step_probe.exe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
RECORD = CRATE / "validation" / "topology_decisions.json"
FIXTURE = CRATE / "target" / "verify_training_stack" / "topology_fixture"

# The geometries the record is audited against, one probe run each. Their keys differ
# in every field the autotuners key a decision on, and the variants do not trade off
# the same way at any two of them: width 64 with ranks 8/8/8 gives the sliced kernel a
# single 64-column slice (the split loses and the 2% margin keeps the one-row kernel),
# width 128 gives it two (the split wins), batch 3 exercises the unpaired-tail key, and
# batch 32 exercises sixteen row pairs. The ladder named `width128-batch2` before this
# set existed, which is why it is still the second entry.
REPLAY_GEOMETRIES = (
    ("width64-batch2", ["--batch", "2", "--tokens", "8", "--hidden", "64",
                        "--context-dim", "64", "--steps", "2", "--warmup", "0"]),
    ("width128-batch2", ["--batch", "2", "--tokens", "8", "--steps", "2", "--warmup", "0"]),
    ("width128-batch3-tail", ["--batch", "3", "--tokens", "8", "--steps", "2",
                              "--warmup", "0"]),
    ("width128-batch32", ["--batch", "32", "--tokens", "8", "--steps", "1",
                          "--warmup", "0"]),
)

RECORD_ENV = "HIERARCHOS_RWKV_LOW_RANK_TOPOLOGY_RECORD"
# A forced decision is not a measurement, and neither is a disabled autotuner, so a
# replay strips both and always records what the device actually measured.
UNMEASURED_ENV = (
    "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY",
    "HIERARCHOS_RWKV_LOW_RANK_DISABLE_LN_FORWARD_AUTOTUNE",
    RECORD_ENV,
)

KEY_FIELDS = (
    "family",
    "device",
    "subgroup_size",
    "width",
    "w_rank",
    "a_rank",
    "g_rank",
    "batch_pairs",
    "has_unpaired_tail",
)

RECORD_NOTE = (
    "Measured LN1/low-rank kernel-topology decisions, one entry per (family, device, "
    "geometry). Written only by `validation/verify_topology_decisions.py --record`, "
    "which merges this machine's replayed decisions and leaves every other device's "
    "entries as they were; `timings_ms` are the medians the choice was made from and "
    "`selected` is the variant the device chose. `margin_fraction` is how much the "
    "fastest candidate beat the structural default on those medians. A variant is only "
    "selected when it beats `structural_default` by the material margin in every one of "
    "`rounds` interleaved rounds: `round_wins` counts the rounds each candidate won, "
    "`decisive_margin_fraction` is the narrowest per-round lead of the selected variant, "
    "and a decision that is not unanimous records the default with the rounds it failed "
    "on instead of a variant the device does not actually reproduce. "
    "`contention_spread` is how far the default's own per-round samples moved "
    "(slowest/fastest) and `contended` says whether that was over the autotuner's "
    "threshold; a contended sweep is not evidence, so it keeps the structural default "
    "and never enters this record. `required_margin_fraction` records the actual "
    "performance admission margin for this adapter/geometry; older entries use 2%. "
    "The measured Ally width-128 FP32 profile requires 15% in every round because "
    "quiet 3-6% wins alternated between topologies inside a 13.7% sample spread. "
    "This changes scheduling admission, not a numerical tolerance."
)

# Fields the unanimous-round rule introduced. A record from before it carries none of
# them, and is audited for the choice alone rather than for evidence it never held.
STABILITY_FIELDS = ("structural_default", "rounds", "round_wins", "contended")

# Mirrors `LAYER_NORM_LOW_RANK_FORWARD_AUTOTUNE_CONTENTION_SPREAD` in
# `src/rwkv_low_rank.rs`: how far the structural default's own per-round samples may
# spread, slowest over fastest, before the rounds are declared too contended to
# adjudicate. Used here only to check that an entry's own numbers agree with its flag;
# the autotuner is what decides.
CONTENTION_SPREAD = 1.15


def decision_key(entry):
    """The identity of a decision: a device, a family and one geometry."""
    missing = [field for field in KEY_FIELDS if field not in entry]
    if missing:
        raise KeyError(f"decision is missing {sorted(missing)}")
    return tuple(entry[field] for field in KEY_FIELDS)


def describe(entry):
    ranks = f"{entry['w_rank']}/{entry['a_rank']}/{entry['g_rank']}"
    tail = " + tail" if entry.get("has_unpaired_tail") else ""
    return (
        f"{entry['family']} device={entry['device']!r} subgroup={entry['subgroup_size']} "
        f"width={entry['width']} ranks={ranks} batch_pairs={entry['batch_pairs']}{tail}"
    )


def read_jsonl(path):
    """Parse the reporter's JSON Lines file into decisions, last one per key."""
    decisions = []
    problems = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as error:
            problems.append(f"{path}: line {number} is not JSON: {error}")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{path}: line {number} is not a JSON object")
            continue
        try:
            decision_key(entry)
        except KeyError as error:
            problems.append(f"{path}: line {number}: {error}")
            continue
        decisions.append(entry)
    merged = {}
    for entry in decisions:
        merged[decision_key(entry)] = entry
    return list(merged.values()), problems


def load_record(path):
    """Return the checked-in decisions and any structural problems with the file."""
    if not Path(path).is_file():
        return [], [f"{path} is missing; create it with --record on a device"]
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != 1:
        return [], [f"{path}: unsupported schema {document.get('schema')!r}"]
    entries = document.get("decisions")
    if not isinstance(entries, list):
        return [], [f"{path}: decisions must be a list"]
    problems = []
    pairs = []
    for entry in entries:
        try:
            pairs.append((decision_key(entry), entry))
        except KeyError as error:
            problems.append(f"{path}: {error}")
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        problems.append(f"{path}: duplicate decisions; only --record should write this file")
    if keys != sorted(keys):
        problems.append(f"{path}: decisions are not sorted; only --record should write this file")
    # The record's own stability evidence has to hold together, not just its choice: a
    # hand-edited entry could claim a variant won a round it lost and nothing else would
    # notice, because the replay only compares `selected`.
    for entry in (entry for _, entry in pairs):
        problems.extend(
            f"{path}: {problem}" for problem in stability_problems(entry, describe(entry))
        )
    return [entry for _, entry in pairs], problems


def stability_problems(entry, label):
    """Check one decision's own stability evidence.

    The rule the record documents is that only a variant which beats the structural
    default in every interleaved round may be selected, so the fields it carries have to
    agree with each other: a non-default `selected` won every round and has a decisive
    margin, and a kept default won none. The contention flag is part of that: a sweep the
    autotuner called contended keeps the structural default, and an entry whose spread
    contradicts its own flag was not written by the autotuner. A record written before
    those fields existed is not a problem here -- it is audited for the choice, which is
    what it holds.
    """
    problems = []
    if any(field not in entry for field in STABILITY_FIELDS):
        return problems
    selected = entry["selected"]
    default = entry["structural_default"]
    rounds = entry["rounds"]
    wins = entry["round_wins"]
    required_margin = entry.get("required_margin_fraction", 0.02)
    if not isinstance(required_margin, (int, float)) or not 0.0 < required_margin < 1.0:
        problems.append(f"{label}: invalid performance admission margin {required_margin!r}")
    if selected not in wins:
        return [
            f"{label}: selected {selected!r} is not among the measured candidates "
            f"{sorted(wins)}"
        ]
    spread = entry.get("contention_spread")
    if entry.get("contended"):
        if selected != default:
            problems.append(
                f"{label}: {selected} is selected from a contended sweep; a sweep whose "
                f"default samples spread more than {CONTENTION_SPREAD}x is not evidence, "
                f"so it keeps {default!r}"
            )
        if spread is not None and spread <= CONTENTION_SPREAD:
            problems.append(
                f"{label}: records contended with a spread of {spread}x, under the "
                f"{CONTENTION_SPREAD}x threshold"
            )
    elif spread is not None and spread > CONTENTION_SPREAD:
        problems.append(
            f"{label}: records a spread of {spread}x over the {CONTENTION_SPREAD}x "
            "contention threshold but is not marked contended"
        )
    if selected != default:
        if rounds <= 0 or wins.get(selected, 0) != rounds:
            problems.append(
                f"{label}: {selected} is selected after winning only "
                f"{wins.get(selected, 0)}/{rounds} round(s); a variant may displace the "
                f"default {default!r} only by beating it in every round"
            )
        elif not entry.get("decisive_margin_fraction"):
            problems.append(
                f"{label}: {selected} won every round but records no decisive margin"
            )
        elif (isinstance(required_margin, (int, float))
              and entry["decisive_margin_fraction"] < required_margin):
            problems.append(f"{label}: decisive margin is below the recorded performance admission margin")
    elif wins.get(selected, 0):
        problems.append(
            f"{label}: kept {selected} as the default yet records it winning "
            f"{wins.get(selected)} round(s)"
        )
    return problems


def compare(replay, recorded, device):
    """Audit one replay against the record. Returns `(problems, notes)`."""
    problems = []
    notes = []
    by_key = {decision_key(entry): entry for entry in recorded}
    measured = set()
    for entry in replay:
        key = decision_key(entry)
        measured.add(key)
        label = describe(entry)
        known = by_key.get(key)
        if known is None:
            if entry.get("contended"):
                notes.append(
                    f"{label}: this sweep was contended (spread "
                    f"{entry.get('contention_spread', 0.0):.2f}x), which cannot record a "
                    "decision; re-run on a quiet device with --record"
                )
            else:
                problems.append(f"{label}: measured here but not recorded; run with --record")
            continue
        problems.extend(stability_problems(entry, label))
        if known.get("required_margin_fraction", 0.02) != entry.get("required_margin_fraction", 0.02):
            problems.append(f"{label}: performance admission margin changed; re-record its measurements")
            continue
        if entry.get("contended"):
            notes.append(
                f"{label}: contended sweep (spread "
                f"{entry.get('contention_spread', 0.0):.2f}x, over the "
                f"{CONTENTION_SPREAD}x threshold), so this run did not adjudicate; the "
                f"recorded {known['selected']} stands"
            )
            continue
        if known["selected"] != entry["selected"]:
            problems.append(
                f"{label}: recorded {known['selected']} but this device now selects "
                f"{entry['selected']}"
            )
            continue
        # The variants a geometry measures are part of the decision: a candidate that
        # stops being timed is a variant this device no longer exercises there, which is
        # not something a matching `selected` should be able to hide.
        measured_here = entry.get("timings_ms", {})
        for variant in sorted(known.get("timings_ms", {})):
            if variant not in measured_here:
                problems.append(
                    f"{label}: {variant} was measured when this decision was recorded and "
                    "this run did not time it"
                )
        for variant in sorted(measured_here):
            if variant not in known.get("timings_ms", {}):
                notes.append(f"{label}: {variant} is newly measured at this geometry")
        notes.append(f"{label}: {entry['selected']} reproduced")
        recorded_timings = known.get("timings_ms", {})
        if entry["selected"] == entry.get("structural_default") and entry.get(
            "margin_fraction", 0.0
        ) > 0.0:
            wins = entry.get("round_wins", {})
            notes.append(
                f"{label}: kept {entry['selected']} against a "
                f"{entry['margin_fraction'] * 100:.1f}% median lead that did not hold in "
                f"every round (round wins: "
                + ", ".join(f"{variant} {count}" for variant, count in sorted(wins.items()))
                + ")"
            )
        for variant, observed_ms in sorted(entry.get("timings_ms", {}).items()):
            before = recorded_timings.get(variant)
            if not before:
                continue
            ratio = observed_ms / before
            if ratio < 0.8 or ratio > 1.25:
                notes.append(
                    f"{label}: {variant} {observed_ms:.5f} ms vs recorded {before:.5f} ms "
                    f"({ratio:.2f}x)"
                )
    for entry in recorded:
        key = decision_key(entry)
        if key in measured:
            continue
        if entry["device"] != device:
            continue
        notes.append(
            f"{describe(entry)}: recorded but this fixture does not exercise it; "
            "re-record it with its own probe arguments if it changed"
        )
    others = {
        entry["device"] for entry in recorded if entry["device"] != device
    }
    if others:
        notes.append(
            "{} recorded decision(s) for other device(s) are not replayed here: {}".format(
                sum(1 for entry in recorded if entry["device"] != device),
                ", ".join(sorted(others)),
            )
        )
    return problems, notes


def recordable(entries):
    """Split a replay into the decisions a record may hold and the contended ones.

    A sweep taken while something else was using the GPU is not evidence about the
    kernels, so it never reaches the record: writing it would turn another process's
    timing into this device's decision, which is exactly what a replay on a loaded
    machine would then dutifully reproduce as "recorded".
    """
    quiet = [entry for entry in entries if not entry.get("contended")]
    contended = [entry for entry in entries if entry.get("contended")]
    return quiet, contended


def merge(recorded, replay):
    """Return the record's decisions with this replay's entries replaced and sorted."""
    merged = {decision_key(entry): entry for entry in recorded}
    for entry in replay:
        merged[decision_key(entry)] = entry
    return [merged[key] for key in sorted(merged)]


def replay(probe, name, arguments, timeout):
    """Run one geometry with the reporter pointed at scratch and parse what it measured."""
    geometry = FIXTURE / name
    if geometry.exists():
        shutil.rmtree(geometry)
    geometry.mkdir(parents=True, exist_ok=True)
    record_file = geometry / "decisions.jsonl"
    environment = dict(os.environ)
    for name in UNMEASURED_ENV:
        environment.pop(name, None)
    environment[RECORD_ENV] = str(record_file)
    completed = subprocess.run(
        [str(probe), "--model-dir", str(geometry / "model"), *arguments],
        cwd=CRATE,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{name}: probe exit {completed.returncode}: "
            f"{(completed.stdout + completed.stderr).strip()[-3000:]}"
        )
    if not record_file.is_file():
        return [], []
    return read_jsonl(record_file)


def geometry_report(measured, geometries):
    """Per-geometry coverage, and the failure a geometry that measured nothing is.

    Every geometry in the replay is a promise that this device chooses a kernel at that
    shape, and the record is only as wide as the geometries that produced it. A shape
    that stops measuring -- the fused producer went away, ranks grew past the kernel's
    limit, a device lost the storage bindings -- would otherwise shrink the audited
    coverage silently while the step kept passing, so it fails here instead.
    """
    notes = []
    problems = []
    for name, _ in geometries:
        entries = measured.get(name, [])
        if not entries:
            problems.append(
                f"{name}: the replay measured no decision, so this geometry is a claim "
                "the device no longer supports; fix why the producer is not fused there "
                "or drop the geometry from REPLAY_GEOMETRIES"
            )
            continue
        notes.append(
            f"{name}: "
            + "; ".join(f"{describe(entry)} -> {entry['selected']}" for entry in entries)
        )
    return notes, problems


def default_probe():
    for profile in ("debug", "release"):
        for name in ("training_step_probe", "training_step_probe.exe"):
            candidate = CRATE / "target" / profile / name
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "no training_step_probe binary found; build it with "
        "`cargo build --bin training_step_probe`"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, help="probe binary (defaults to target/{debug,release})")
    parser.add_argument("--record", action="store_true", help="merge this run's decisions into the record")
    parser.add_argument("--record-file", type=Path, default=RECORD)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    probe = args.probe or default_probe()
    probe = probe.resolve()
    recorded, record_problems = load_record(args.record_file)
    if args.record:
        # Recording is how the record comes into existence, so an absent file is not a
        # problem for that run; every other structural problem still is.
        record_problems = [
            problem for problem in record_problems if "is missing" not in problem
        ]
    problems = list(record_problems)
    notes = []

    measured_by_geometry = {}
    for name, arguments in REPLAY_GEOMETRIES:
        entries, replay_problems = replay(probe, name, arguments, args.timeout)
        measured_by_geometry[name] = entries
        problems.extend(replay_problems)
    coverage_notes, coverage_problems = geometry_report(measured_by_geometry, REPLAY_GEOMETRIES)
    notes.extend(coverage_notes)
    problems.extend(coverage_problems)

    # One decision per key: a key measured at two geometries is one entry, and the last
    # geometry that measured it wins, exactly as the reporter's own file does.
    measured = {}
    for entries in measured_by_geometry.values():
        for entry in entries:
            measured[decision_key(entry)] = entry
    measured = list(measured.values())
    device = measured[0]["device"] if measured else f"(unmeasured: {probe.name})"

    if args.record:
        quiet, contended = recordable(measured)
        if contended:
            notes.append(
                f"{len(contended)} decision(s) were measured under contention and were "
                "not recorded; the record holds only sweeps that measured one machine"
            )
        if not quiet:
            notes.append("nothing was measured on this run, so the record is unchanged")
        else:
            merged = merge(recorded, quiet)
            document = (
                json.loads(args.record_file.read_text(encoding="utf-8"))
                if args.record_file.is_file()
                else {}
            )
            document["schema"] = 1
            document["note"] = RECORD_NOTE
            document["decisions"] = merged
            args.record_file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            notes.append(
                f"recorded {len(quiet)} decision(s) for {device!r}; "
                f"the record now holds {len(merged)}"
            )
    elif measured:
        # An empty replay is already a coverage failure above, so there is nothing to
        # compare in that case and no second message for it.
        replay_problems, replay_notes = compare(measured, recorded, device)
        problems.extend(replay_problems)
        notes.extend(replay_notes)

    report = {
        "device": device,
        "probe": str(probe),
        "record": str(args.record_file),
        "measured": measured,
        "geometries": measured_by_geometry,
        "notes": notes,
        "problems": problems,
        "pass": not problems,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for note in notes:
        print(f"note  {note}")
    for problem in problems:
        print(f"FAIL  {problem}")
    if problems:
        print(f"FAIL: {len(problems)} topology decision problem(s) on {device!r}")
        return 1
    print(
        f"PASS: {len(measured)} measured decision(s) match the record on {device!r}"
        if measured
        else f"PASS: nothing to replay on {device!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

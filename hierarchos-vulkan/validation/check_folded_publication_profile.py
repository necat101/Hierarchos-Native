#!/usr/bin/env python3
"""Check the committed folded-publication profiles against the claim they support.

`bench_rwkv_tbptt_packed_chains_bench --profile-folded` (driven by
`bench_rwkv_tbptt_packed_chains.py --profile-folded`) splits the cost of the
fully folded state-backward rung into the two things the fold adds: the arrival
counters with their device-scope barriers, and the batch-order reduction the
last-arriving workgroup performs. The committed profiles are what
`validation/FUSED_FORWARD_PRODUCER_AUDIT.md` quotes when it says *neither*
component is the limit on the recorded adapter, so they are worth exactly as much
as the check that reads them. This guard is that check, on the CPU and without a
GPU:

* the header names the adapter, the geometry and the sampling the profile claims,
  including a floor on the number of rotated samples: a profile timed with a
  handful of samples cannot resolve anything and must not be cited as noise;
* every geometry in the sweep appears once, in order, and batch 1 carries no
  phantom-arrival leg -- that leg removes the reduction, and batch 1 has none;
* the three timed legs are the three fusion depths. Their recorded schedules must
  resolve to `rkv-add3+key+reduce`, `rkv-add3-key+reduce` and
  `rkv-add3-key-reduce`, so a profile cannot silently time the same graph twice
  and call the difference a component;
* the gates the legs cleared before timing are recorded and must hold: the clean
  fold publishes the staged arm's `grad_k_k`/`grad_k_a` exactly, the staged
  magnitude is non-zero (or that equality is zero-vs-zero), and the
  phantom-arrival leg fails the same gate;
* every reported number is re-derived from the legs it claims to summarise --
  medians, ms per step, the two component deltas, the ratio, and the resolution
  (the widest leg's interquartile range). A hand-edited delta or a resolution
  that does not match its legs fails here;
* each component's `*_beyond_resolution` flag must agree with its own delta, and
  no component may be flagged. This is the claim under test: the audit says both
  components sit inside the run's noise. A profile where one becomes visible is
  a real result, and it fails this check until the audit's conclusion and this
  file are changed in the same diff.

`--compare` applies the same rules to a freshly recorded profile and refuses a
run whose fold leads beyond its own resolution, or whose components become
visible; timings that merely differ are reported as notes, because two runs on
one adapter are not expected to agree to the microsecond.

Usage:
    python hierarchos-vulkan/validation/check_folded_publication_profile.py
    python hierarchos-vulkan/validation/check_folded_publication_profile.py --profile fp16-storage
    python hierarchos-vulkan/validation/check_folded_publication_profile.py \\
        --compare target/folded_profile_fp32.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"

# The profiles the audit quotes, and the storage policy each was recorded under.
# A policy-less entry is the FP32 run; the other names its own policy so a profile
# recorded under the packed mirrors cannot be cited as the FP32 measurement.
PROFILES = (
    ("fp32", VALIDATION / "tbptt_folded_publication_profile.json", None),
    (
        "fp16-storage",
        VALIDATION / "tbptt_folded_publication_profile_fp16.json",
        "fp16-storage",
    ),
)

# The profile's identity, and the sampling floors it must meet. The floors live
# here rather than in the artifact so that editing the committed JSON cannot
# lower them; the audit's "inside the resolution" conclusion rests on them.
RECORDED_DEVICE = "Intel(R) HD Graphics 520"
PROFILE_KIND = "folded-publication"
SWEEP = (1, 4, 8, 16, 32)
STEPS = 8
MIN_SAMPLES = 40
MIN_WARMUP = 3

# The three fusion depths, as the schedule labels name them: `+` is a stage that
# still dispatches and `-` is one folded into the state-backward kernel.
DEPTHS = {
    "staged": "rkv-add3+key+reduce",
    "fused": "rkv-add3-key+reduce",
    "folded": "rkv-add3-key-reduce",
}
PHANTOM = "folded-without-reduction"


class Problem(Exception):
    """A check failed; the message is the report line."""


@dataclass(frozen=True)
class Spec:
    name: str
    path: Path
    parameter_storage: str | None
    # A profile is one adapter's measurement, so the adapter is part of what the
    # entry asserts. A second vendor's profile is a new entry with its own device,
    # not a re-recorded one that silently replaces this device's numbers.
    device: str = RECORDED_DEVICE


def specs() -> tuple[Spec, ...]:
    return tuple(Spec(name, path, policy) for name, path, policy in PROFILES)


def spec_named(name: str) -> Spec:
    for spec in specs():
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in specs())
    raise Problem(f"no profile named {name!r}; the guard knows {known}")


def load(path: Path) -> dict:
    if not path.exists():
        raise Problem(
            f"missing {path.name}; record it with "
            "bench_rwkv_tbptt_packed_chains.py --profile-folded --record"
        )
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise Problem(f"{path.name} is not valid JSON: {err}") from err
    if not isinstance(profile, dict):
        raise Problem(f"{path.name} holds {type(profile).__name__}, not a profile object")
    return profile


def close(left: float, right: float, *, tol: float = 1.0e-9) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def check_header(profile: dict, spec: Spec) -> None:
    if profile.get("result") != "pass":
        raise Problem(f"{spec.path.name} recorded result {profile.get('result')!r}, not 'pass'")
    if profile.get("profile") != PROFILE_KIND:
        raise Problem(
            f"{spec.path.name} records profile {profile.get('profile')!r}, not "
            f"{PROFILE_KIND!r}; this is a different measurement"
        )
    if profile.get("parameter_storage") != spec.parameter_storage:
        raise Problem(
            f"{spec.path.name} records parameter_storage "
            f"{profile.get('parameter_storage')!r}, the {spec.name} profile is "
            f"{spec.parameter_storage!r}"
        )
    rows = profile.get("batches")
    if not isinstance(rows, list) or not rows:
        raise Problem(f"{spec.path.name} holds no batches")


def depths(row: dict) -> dict[str, dict]:
    """The three timed legs, keyed by depth, after checking the legs are distinct."""
    legs = row.get("legs")
    if not isinstance(legs, list):
        raise Problem(f"batch {row.get('batch')}: no legs recorded")
    labelled = {leg.get("label"): leg for leg in legs}
    if len(labelled) != len(legs):
        raise Problem(f"batch {row.get('batch')}: a leg label is repeated")
    missing = [depth for depth in DEPTHS if depth not in labelled]
    if missing:
        raise Problem(f"batch {row.get('batch')}: no {', '.join(missing)} leg")
    extra = sorted(set(labelled) - set(DEPTHS) - {PHANTOM})
    if extra:
        raise Problem(f"batch {row.get('batch')}: unexpected leg(s) {extra}")
    schedules = {
        depth: row.get(f"{depth}_schedule") for depth in DEPTHS
    }
    for depth, prefix in DEPTHS.items():
        schedule = schedules[depth]
        if not isinstance(schedule, str) or not schedule.startswith(prefix):
            raise Problem(
                f"batch {row.get('batch')}: the {depth} leg pinned {schedule!r}, "
                f"which is not the {prefix!r} depth"
            )
    if len(set(schedules.values())) != len(DEPTHS):
        raise Problem(
            f"batch {row.get('batch')}: the three depths resolved to "
            f"{sorted(set(schedules.values()))}, so two legs timed the same graph"
        )
    return labelled


def check_leg(leg: dict, batch: int, steps: int) -> None:
    label = leg["label"]
    values = {key: leg.get(key) for key in ("median_ms", "ms_per_step", "min_ms", "max_ms", "iqr_ms")}
    if any(not isinstance(value, (int, float)) for value in values.values()):
        raise Problem(f"batch {batch}: the {label} leg is missing a timing")
    median, per_step, low, high, iqr = (
        values["median_ms"],
        values["ms_per_step"],
        values["min_ms"],
        values["max_ms"],
        values["iqr_ms"],
    )
    if not low <= median <= high:
        raise Problem(
            f"batch {batch}: the {label} leg's median {median} is outside its own range "
            f"[{low}, {high}]"
        )
    if iqr < 0 or iqr > high - low:
        raise Problem(
            f"batch {batch}: the {label} leg's interquartile range {iqr} is not in "
            f"[0, {high - low}]"
        )
    if not close(per_step, median / steps):
        raise Problem(
            f"batch {batch}: the {label} leg's ms per step {per_step} is not its median "
            f"{median} over {steps} steps"
        )


def check_row(row: dict, batch: int, spec: Spec) -> dict:
    """One geometry's row: geometry, legs, arithmetic, flags, gates, and the claim."""
    if row.get("batch") != batch:
        raise Problem(f"{spec.path.name}: expected batch {batch}, recorded {row.get('batch')!r}")
    if row.get("device") != spec.device:
        raise Problem(
            f"batch {batch}: recorded on {row.get('device')!r}, not the adapter this "
            f"profile's entry names ({spec.device!r}); a second vendor's measurement is "
            "a new entry in PROFILES, not a re-recorded one"
        )
    if row.get("steps") != STEPS:
        raise Problem(f"batch {batch}: recorded {row.get('steps')!r} steps, expected {STEPS}")
    samples, warmup = row.get("samples"), row.get("warmup")
    if not isinstance(samples, int) or samples < MIN_SAMPLES:
        raise Problem(
            f"batch {batch}: {samples!r} rotated samples, below the {MIN_SAMPLES} this "
            "profile needs to call a delta noise"
        )
    if not isinstance(warmup, int) or warmup < MIN_WARMUP:
        raise Problem(f"batch {batch}: {warmup!r} warmup rounds, below the {MIN_WARMUP} floor")
    for key in ("width", "head_size"):
        if not isinstance(row.get(key), int) or row[key] <= 0:
            raise Problem(f"batch {batch}: {key} is {row.get(key)!r}")

    legs = depths(row)
    for leg in legs.values():
        check_leg(leg, batch, STEPS)
    if (PHANTOM in legs) != (batch > 1):
        raise Problem(
            f"batch {batch}: the phantom-arrival leg is "
            f"{'missing' if batch > 1 else 'present'}; batch 1 has no counter or reduction "
            "to remove and every other batch must be split"
        )

    staged, fused, folded = (legs[depth]["median_ms"] for depth in ("staged", "fused", "folded"))
    without = legs[PHANTOM]["median_ms"] if PHANTOM in legs else None

    # The reported numbers must be the legs' own numbers, not a separate claim.
    if not close(row.get("folded_over_fused"), folded / fused, tol=1.0e-12):
        raise Problem(
            f"batch {batch}: folded/fused {row.get('folded_over_fused')} is not "
            f"{folded} / {fused}"
        )
    if not close(row.get("folded_minus_fused_ms"), folded - fused):
        raise Problem(
            f"batch {batch}: folded-minus-fused {row.get('folded_minus_fused_ms')} is not "
            f"{folded} - {fused}"
        )
    resolution = row.get("resolution_ms")
    if not isinstance(resolution, (int, float)) or resolution <= 0:
        raise Problem(f"batch {batch}: resolution {resolution!r} is not a positive time")
    widest_iqr = max(leg["iqr_ms"] for leg in legs.values())
    if not close(resolution, widest_iqr):
        raise Problem(
            f"batch {batch}: resolution {resolution} is not the widest leg's interquartile "
            f"range {widest_iqr}; the flags below are relative to it"
        )
    widest_half_range = max((leg["max_ms"] - leg["min_ms"]) / 2.0 for leg in legs.values())
    if not close(row.get("range_resolution_ms"), widest_half_range):
        raise Problem(
            f"batch {batch}: range resolution {row.get('range_resolution_ms')} is not the "
            f"widest half-range {widest_half_range}"
        )

    if without is not None:
        tail = row.get("folded_minus_without_reduction_ms")
        counters = row.get("without_reduction_minus_fused_ms")
        if not close(tail, folded - without):
            raise Problem(
                f"batch {batch}: reduction tail {tail} is not {folded} - {without}"
            )
        if not close(counters, without - fused):
            raise Problem(
                f"batch {batch}: counter delta {counters} is not {without} - {fused}"
            )
        expected_tail = abs(folded - without) > resolution
        expected_counters = abs(without - fused) > resolution
        if row.get("reduction_tail_beyond_resolution") is not expected_tail:
            raise Problem(
                f"batch {batch}: the tail is flagged "
                f"{row.get('reduction_tail_beyond_resolution')!r}, but |{folded} - {without}| "
                f"against a {resolution} resolution says {expected_tail}"
            )
        if row.get("counters_beyond_resolution") is not expected_counters:
            raise Problem(
                f"batch {batch}: the counters are flagged "
                f"{row.get('counters_beyond_resolution')!r}, but |{without} - {fused}| "
                f"against a {resolution} resolution says {expected_counters}"
            )

    expected_lead = fused - folded > resolution
    if row.get("folded_beats_fused_beyond_resolution") is not expected_lead:
        raise Problem(
            f"batch {batch}: the fold's lead is flagged "
            f"{row.get('folded_beats_fused_beyond_resolution')!r}, but {fused} - {folded} "
            f"against a {resolution} resolution says {expected_lead}"
        )

    # The gates the legs cleared before they were timed. Without these the timings
    # would be two graphs' timings, not the fold's.
    atol = row.get("key_parameter_atol")
    if atol != 0.0:
        raise Problem(
            f"batch {batch}: the key-parameter allowance is {atol!r}; the fold must publish "
            "the two-dispatch reduction's gradients exactly"
        )
    magnitude = row.get("staged_key_parameter_max_abs")
    if not isinstance(magnitude, (int, float)) or magnitude <= 0:
        raise Problem(
            f"batch {batch}: the staged arm's grad_k_k/grad_k_a are {magnitude!r}, so the "
            "fold's agreement with them would prove nothing"
        )
    clean = row.get("folded_key_parameter_max_abs")
    if not isinstance(clean, (int, float)) or clean > atol:
        raise Problem(
            f"batch {batch}: the clean fold moved grad_k_k/grad_k_a by {clean!r} away from "
            "the two-dispatch chain"
        )
    if without is not None:
        phantom = row.get("without_reduction_key_parameter_max_abs")
        if not isinstance(phantom, (int, float)) or phantom <= atol:
            raise Problem(
                f"batch {batch}: the phantom-arrival leg published the key-parameter "
                f"gradients anyway ({phantom!r}); its timing is not the fold without its "
                "reduction"
            )
        if phantom <= clean:
            raise Problem(
                f"batch {batch}: the phantom-arrival leg is no further from the staged arm "
                f"({phantom}) than the clean fold ({clean}), so the control did nothing"
            )
    elif row.get("without_reduction_key_parameter_max_abs") is not None:
        raise Problem(
            f"batch {batch}: a phantom-arrival key-parameter delta is recorded without that leg"
        )

    # The claim the audit makes with this file.
    if expected_lead:
        raise Problem(
            f"batch {batch}: the folded rung now leads by {folded} vs {fused} "
            f"({fused - folded:.3f} ms), beyond the {resolution:.3f} ms resolution. That is "
            "a real result: re-measure, then change this guard and the audit's conclusion "
            "in the same diff"
        )
    # The two components are separated by the phantom leg's own timing, so moving
    # that leg can surface both at once; name every component that did.
    visible = []
    if row.get("reduction_tail_beyond_resolution") is True:
        visible.append("batch-order reduction")
    if row.get("counters_beyond_resolution") is True:
        visible.append("arrival counters")
    if visible:
        raise Problem(
            f"batch {batch}: the {' and the '.join(visible)} now resolvable, so the audit's "
            "'neither component is the limit' conclusion no longer holds for this geometry"
        )
    return legs


def check_profile(profile: dict, spec: Spec) -> list[dict]:
    check_header(profile, spec)
    rows = profile["batches"]
    batches = [row.get("batch") for row in rows]
    if tuple(batches) != SWEEP:
        raise Problem(
            f"{spec.path.name} records batches {batches}; the sweep is {list(SWEEP)}. A "
            "geometry dropped from the profile is a geometry the claim was not measured at"
        )
    for batch, row in zip(SWEEP, rows):
        check_row(row, batch, spec)
    geometry = {(row.get("width"), row.get("head_size")) for row in rows}
    if len(geometry) != 1:
        raise Problem(f"{spec.path.name} mixes geometries {sorted(geometry)}")
    return rows


def summary_line(row: dict) -> str:
    legs = {leg["label"]: leg["median_ms"] for leg in row["legs"]}
    line = (
        f"batch {row['batch']:>2}: staged {legs['staged']:8.3f} ms  "
        f"fused {legs['fused']:8.3f} ms  folded {legs['folded']:8.3f} ms  "
        f"folded/fused {row['folded_over_fused']:.4f}  "
        f"resolution +/-{row['resolution_ms']:.3f} ms"
    )
    if PHANTOM in legs:
        line += (
            f"  tail {row['folded_minus_without_reduction_ms']:+.3f} ms  "
            f"counters {row['without_reduction_minus_fused_ms']:+.3f} ms"
        )
    return line


def check_compare(fresh: dict, committed: dict, spec: Spec) -> list[str]:
    """A fresh profile against the committed one.

    The fresh run gets every check the committed file gets, so a run whose fold
    leads beyond its own resolution fails here as well. Timings differ between
    runs, so deltas that moved are reported rather than rejected.
    """
    fresh_rows = check_profile(fresh, spec)
    committed_rows = check_profile(committed, spec)
    notes = []
    for fresh_row, committed_row in zip(fresh_rows, committed_rows):
        batch = fresh_row["batch"]
        if fresh_row["folded_beats_fused_beyond_resolution"]:
            raise Problem(f"batch {batch}: the fresh run's fold leads beyond its resolution")
        drift = fresh_row["folded_over_fused"] - committed_row["folded_over_fused"]
        if abs(drift) > 0.01:
            notes.append(
                f"batch {batch}: folded/fused moved {drift:+.3f} against the committed "
                f"{committed_row['folded_over_fused']:.4f}; re-record if this is the new "
                "measurement"
            )
        if fresh_row["resolution_ms"] > committed_row["resolution_ms"] * 2:
            notes.append(
                f"batch {batch}: the fresh run's resolution is "
                f"{fresh_row['resolution_ms']:.3f} ms against the committed "
                f"{committed_row['resolution_ms']:.3f} ms; it resolved less than the "
                "committed run"
            )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        action="append",
        metavar="NAME",
        help="check only this committed profile (repeatable; default both)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="a freshly recorded profile to compare against the committed one",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="check this profile file instead of the committed one (with --compare's name)",
    )
    args = parser.parse_args(argv)
    try:
        selected = [spec_named(name) for name in args.profile] if args.profile else list(specs())
        if args.profile_file is not None and len(selected) != 1:
            raise Problem("--profile-file needs exactly one --profile name")
        checked = 0
        for spec in selected:
            path = args.profile_file or spec.path
            profile = load(path)
            rows = check_profile(profile, spec)
            for row in rows:
                print(f"{spec.name:>12}  {summary_line(row)}")
            if args.compare is not None:
                fresh = load(args.compare)
                for line in check_compare(fresh, profile, spec):
                    print(f"note: {line}")
            checked += 1
        print(
            f"PASS: {checked} folded-publication profile(s) hold their arithmetic, their "
            "gates and the claim that neither component is resolvable"
        )
    except Problem as err:
        print(f"FAIL: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

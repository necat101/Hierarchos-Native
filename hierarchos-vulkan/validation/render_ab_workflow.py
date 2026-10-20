#!/usr/bin/env python3
"""Render the A/B workflow's per-producer steps from the route guard's spec table.

`validation/check_measured_producer_routes.py` already owns the contract each
measured producer has to satisfy: its driver, the sweep it must dispatch, the
receipt it writes, and the guard invocation that re-checks it against the record
it feeds. Keeping a second copy of that list in `.github/workflows/time-mix-ab.yml`
is how a producer's sweep or its guard step goes missing -- the workflow still
looks complete while one receipt is no longer re-checked on hardware.

This module renders the per-job step blocks, the receipt-contract test matrix
and the artifact paths from `ALL_SPECS`, the same table the guard runs from. The
hand-written parts are only the triggers, runner setup and the phase-profile step.
A new producer added to that table therefore appears in every job its `jobs`
field names, with its full sweep, its guard step, its receipt in the artifact
list and its receipt-contract module in the test matrix.

Usage:
    python hierarchos-vulkan/validation/render_ab_workflow.py --check
    python hierarchos-vulkan/validation/render_ab_workflow.py --write
    python hierarchos-vulkan/validation/render_ab_workflow.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_measured_producer_routes as guard  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
WORKFLOW = WORKSPACE / ".github" / "workflows" / "time-mix-ab.yml"
GUARD_SCRIPT = "hierarchos-vulkan/validation/check_measured_producer_routes.py"

JOBS = ("producer-ab", "cross-adapter-ab")
# The cross-adapter job runs the same sweep on a second vendor, so its step
# names say which adapter produced the receipt.
JOB_SUFFIX = {"producer-ab": "", "cross-adapter-ab": " (AMD adapter)"}
# Only the producer job measures "the shipped default"; the cross-adapter job
# records a fresh adapter's receipts, which have no recorded route yet.
PRODUCER_JOB_TAIL = " on the shipped default"


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def fold(words: list[str], width: int = 100) -> list[str]:
    """Wrap one argument list into lines a reviewer can diff, never mid-flag."""
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def step(name: str, command: list[str]) -> list[str]:
    lines = [f"      - name: {name}", "        run: >"]
    lines.extend(f"          {line}" for line in command)
    return lines


def bench_command(spec: guard.Producer | guard.CompositionConsumer) -> list[str]:
    """The driver invocation, with the sweep expanded from the spec."""
    lines = [f"python {spec.driver}"]
    args = list(spec.bench_args)
    if spec.step_kind != "composition":
        args = [word for batch in spec.sweep for word in ("--batch", str(batch))] + args
    lines.extend(fold(args))
    lines.append(f"--record {relative(spec.receipt)}")
    return lines


def spec_steps(spec: guard.Producer | guard.CompositionConsumer, job: str) -> list[str]:
    suffix = JOB_SUFFIX[job]
    if spec.step_kind == "composition":
        name = f"{spec.label}{suffix}"
    else:
        tail = PRODUCER_JOB_TAIL if job == "producer-ab" else ""
        name = f"{spec.label} A/B{tail}{suffix}"
    lines = step(name, bench_command(spec))
    lines.append("")
    lines.extend(
        step(
            f"{spec.guard_label}{suffix}",
            [f"python {GUARD_SCRIPT}", f"--producer {spec.name}"],
        )
    )
    return lines


def validate_specs(specs: tuple) -> None:
    """Refuse to render a spec the workflow cannot invoke or identify."""
    problems = []
    for spec in specs:
        if not spec.label or not spec.guard_label:
            problems.append(f"{spec.name}: label and guard_label are required")
        if not spec.driver.startswith("hierarchos-vulkan/validation/"):
            problems.append(f"{spec.name}: driver must name a validation script")
        if not spec.jobs:
            problems.append(f"{spec.name}: jobs is empty, so the sweep never runs")
        if not spec.test_module:
            problems.append(f"{spec.name}: test_module is required for the generated matrix")
        unknown = set(spec.jobs) - set(JOBS)
        if unknown:
            problems.append(f"{spec.name}: unknown jobs {sorted(unknown)}")
    if problems:
        raise SystemExit("cannot render the A/B workflow: " + "; ".join(problems))


def render_steps(job: str, specs: tuple = guard.ALL_SPECS) -> str:
    """The generated block for one job, one blank line between producer pairs."""
    validate_specs(specs)
    chunks = [
        spec_steps(spec, job)
        for spec in specs
        if job in spec.jobs
    ]
    lines: list[str] = []
    for index, chunk in enumerate(chunks):
        if index:
            lines.append("")
        lines.extend(chunk)
    return "\n".join(lines)


def artifact_paths(job: str, specs: tuple = guard.ALL_SPECS) -> list[str]:
    """Every receipt this job writes, in spec order and without duplicates."""
    paths: list[str] = []
    for spec in specs:
        if job not in spec.jobs:
            continue
        for path in (spec.receipt, *spec.extra_artifacts):
            shown = relative(path)
            if shown not in paths:
                paths.append(shown)
    return paths


def render_artifacts(job: str, specs: tuple = guard.ALL_SPECS) -> str:
    validate_specs(specs)
    return "\n".join(
        f"            {path}" for path in artifact_paths(job, specs)
    )


def test_modules(specs: tuple = guard.ALL_SPECS) -> list[str]:
    """The receipt-contract suites, followed by the guard and renderer suites."""
    modules: list[str] = []
    for spec in specs:
        for module in (spec.test_module, *spec.extra_test_modules):
            if module and module not in modules:
                modules.append(module)
    for module in ("test_measured_producer_routes.py", "test_ab_workflow_rendering.py"):
        if module not in modules:
            modules.append(module)
    return modules


def render_tests(specs: tuple = guard.ALL_SPECS) -> str:
    validate_specs(specs)
    return "\n".join(
        f'          python -m unittest discover -s hierarchos-vulkan/validation -p "{module}" -v'
        for module in test_modules(specs)
    )


def markers(tag: str, indent: int) -> tuple[str, str]:
    lead = " " * indent
    return (f"{lead}# BEGIN GENERATED {tag}", f"{lead}# END GENERATED {tag}")


def replace_block(text: str, tag: str, body: str, indent: int = 6) -> str:
    begin, end = markers(tag, indent)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(
            f"{relative(WORKFLOW)} needs exactly one {begin!r} and one {end!r} marker"
        )
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def replace_every_block(text: str, tag: str, body: str, indent: int = 6) -> str:
    """Replace every marker pair with the same body (the test matrix is shared)."""
    begin, end = markers(tag, indent)
    if text.count(begin) < 1 or text.count(begin) != text.count(end):
        raise SystemExit(
            f"{relative(WORKFLOW)} needs matching {begin!r} and {end!r} markers"
        )
    parts = text.split(begin)
    rendered = parts[0]
    for part in parts[1:]:
        _, tail = part.split(end, 1)
        rendered += f"{begin}\n{body}\n{end}{tail}"
    return rendered


def render_workflow(text: str, specs: tuple = guard.ALL_SPECS) -> str:
    for job in JOBS:
        text = replace_block(text, f"STEPS: {job}", render_steps(job, specs))
        text = replace_block(
            text, f"ARTIFACTS: {job}", render_artifacts(job, specs), indent=12
        )
    return replace_every_block(text, "TESTS", render_tests(specs), indent=10)


def first_difference(rendered: str, committed: str) -> str:
    for number, (left, right) in enumerate(
        zip(rendered.splitlines(), committed.splitlines()), start=1
    ):
        if left != right:
            return f"line {number}:\n  rendered:  {left}\n  committed: {right}"
    return f"the committed file has {len(committed.splitlines())} lines, rendered {len(rendered.splitlines())}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the committed workflow is not the rendered one",
    )
    action.add_argument(
        "--write", action="store_true", help="rewrite the committed workflow"
    )
    args = parser.parse_args(argv)

    committed = WORKFLOW.read_text(encoding="utf-8")
    rendered = render_workflow(committed)
    if args.check:
        if rendered == committed:
            print(f"PASS: {relative(WORKFLOW)} matches the guard's spec table")
            return 0
        print(
            f"FAIL: {relative(WORKFLOW)} is stale: {first_difference(rendered, committed)}\n"
            "Run `python hierarchos-vulkan/validation/render_ab_workflow.py --write`.",
            file=sys.stderr,
        )
        return 1
    if args.write:
        WORKFLOW.write_text(rendered, encoding="utf-8")
        print(f"wrote {relative(WORKFLOW)}")
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())

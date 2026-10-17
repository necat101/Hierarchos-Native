#!/usr/bin/env python3
"""Run the whole Hierarchos Vulkan training verification ladder in one pass.

The checks below were previously run by hand, in this order, because each one
constrains a different layer and a change can satisfy one while breaking another:

  1. build the four ladder binaries          - every required harness builds
  2. shader toolchain                        - the pinned releases are on PATH,
                                              installed into `.tools/` when they are
                                              missing, or the exact install steps are
                                              reported as the blocker
  3. binding-access audit (`--check`)        - no shader still declares access that
                                              is provably narrower than its
                                              declaration
  4. shader provenance                       - the checked-in `.spv` files are what
                                              their sources, documented flags and
                                              recorded compiler releases produce
  5. push-constant layouts                   - every kernel's push block agrees
                                              between the committed SPIR-V, the
                                              GLSL it was built from, and the host
                                              struct whose bytes are pushed, so a
                                              stale module cannot silently ignore a
                                              member the host sends
  6. shader oracle parity                    - no portable shader artifact drifts
                                              from the read-only stable baseline;
                                              BLOCKED, never green, on a machine
                                              without that tree
  7. golden digests, three parity harnesses  - kernel outputs are still
                                              bit-identical, checked against
                                              recorded digests instead of
                                              pre-change artifacts, and which
                                              invocations each device exercises
                                              is audited against the checked-in
                                              coverage record, so a case that
                                              stops running fails here
  8. training-step probe                     - the end-to-end step still has the
                                              expected dispatch and barrier counts,
                                              the expected loss, and a finite graph
  9. topology decisions                      - the kernel variants this device's
                                              autotuners select are the ones the
                                              checked-in record documents
 10. full training trajectories              - parameters, Adam moments/clocks,
                                              losses and recurrent adjoints match
                                              their recorded goldens under both
                                              training numerical policies, once for
                                              the shipped choice and once for every
                                              LN1/low-rank topology variant forced
                                              through the env override, so each
                                              variant's bit-exactness is a step
                                              rather than a manual run. The arms and
                                              golden cases this device exercises are
                                              then audited against the checked-in
                                              coverage record, so an arm or a case
                                              that stops being tested fails here
                                              instead of shrinking quietly

The toolchain step exists because the provenance step cannot check anything without
a compiler: a fresh clone used to report provenance as BLOCKED with `glslc must be on
PATH`, which says nothing about what to do. The step now resolves the repository's
`.tools/` install first, runs `validation/install_shader_toolchain.sh` when the tools
are still missing, and reports the pinned releases it ended up using. Pass
`--no-toolchain-install` to only report. When even that cannot produce the pinned
releases the step is BLOCKED with the install commands rather than with a compiler
saying it is not on PATH.

Each step reports PASS, FAIL, or BLOCKED, and the script exits non-zero unless
every step passed. A step that cannot run — missing toolchain, missing binary,
unrecorded device — is a failure, not a pass: the value of the ladder is that green
means verified.

The trajectory step covers two suites, because the training numerical policy selects a
different LN1/low-rank kernel family: `training_state.json` is the FP32 trajectory that
is pinned to the pre-optimization baseline, and `training_state_fp16.json` is recorded
under `HIERARCHOS_VULKAN_TRAINING_PRECISION=fp16-storage-fp32-compute` to pin the packed
FP16 producer's variants to each other. It is the ladder's longest step by far -- every
arm replays every case of both goldens.

The probe is compared against `validation/training_step_probe/step_expectation.json`.
It runs on a fresh model package under `target/`, because an existing package keeps
advancing through AdamW and every run would then measure a different point in
training. The dispatch count is structural and must match, the barrier count is an
upper bound so that an elision improvement does not fail the ladder, and the loss is
compared bit-for-bit per device name, since bit-identical arithmetic is only
guaranteed within one driver. A device with no recorded loss is reported as
unverified, which fails, and `--record-expectation` writes the entry for the device
in this machine.

Usage:
    python validation/verify_training_stack.py
    python validation/verify_training_stack.py --release
    python validation/verify_training_stack.py --only golden
    python validation/verify_training_stack.py --record-expectation
    python validation/verify_training_stack.py --no-toolchain-install
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from check_shader_provenance import probe

CRATE = Path(__file__).resolve().parents[1]
GOLDEN_DIR = CRATE / "validation" / "kernel_golden"
EXPECTATION = CRATE / "validation" / "training_step_probe" / "step_expectation.json"
SCRATCH = CRATE / "target" / "verify_training_stack"

# The probe is a fixed fixture: these arguments and the expectation go together,
# and the comparison fails if they drift apart.
PROBE_ARGS = ["--batch", "2", "--tokens", "8", "--steps", "2", "--warmup", "0"]

HARNESSES = [
    ("golden dense (matmul)", "matmul_kernel_validation", GOLDEN_DIR / "dense.json"),
    ("golden reduction", "reduction_kernel_validation", GOLDEN_DIR / "reduction.json"),
    ("golden attention", "attention_kernel_cache_check", GOLDEN_DIR / "attention.json"),
]

# Exactly the binaries this ladder runs. `cargo build --bins` would also compile
# every unrelated harness in the crate, so an unrelated bin that does not build
# would make this ladder red for a reason that has nothing to do with the stack it
# verifies.
LADDER_BINS = [harness for _, harness, _ in HARNESSES] + [
    "training_step_probe",
    # The training-state coverage record is audited and merged by the crate's shared record
    # implementation, so the driver needs this tool next to the probe.
    "coverage_record",
]

EXPECTATION_NOTE = (
    "Expectation for validation/verify_training_stack.py. probe_args and "
    "shader_barrier_count_max come from the fixture: the barrier count is a "
    "device-independent upper bound so an elision improvement does not fail the "
    "ladder. mean_loss and dispatch_count are recorded per device and compared for "
    "that device only: bit-identical arithmetic is only guaranteed within one driver, "
    "and the vendor layer selects different kernels on different parts, so the same "
    "fixture does not dispatch the same number of times on every device. Re-record "
    "with --record-expectation after an intended numerical change; that adds this "
    "device's entry and leaves every other device's entry as recorded."
)


class Step:
    """One verification step and its outcome."""

    def __init__(self, name, detail="", ok=False, blocked=False, output=""):
        self.name = name
        self.detail = detail
        self.ok = ok
        self.blocked = blocked
        self.output = output

    @property
    def status(self):
        if self.ok:
            return "PASS"
        return "BLOCKED" if self.blocked else "FAIL"


def run(command):
    """Run a command in the crate directory, returning (exit code, combined output).

    A missing executable is code 127 with the error text rather than an exception,
    so every step still produces a status line.
    """
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(CRATE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as err:
        return 127, f"cannot execute {command[0]}: {err}"
    return completed.returncode, completed.stdout


def last_nonempty(output):
    for line in reversed(output.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def binary(name, profile):
    return CRATE / "target" / profile / (name + (".exe" if os.name == "nt" else ""))


def build(profile, skip):
    if skip:
        return None
    flag = ["--release"] if profile == "release" else []
    binaries = [part for name in LADDER_BINS for part in ("--bin", name)]
    code, output = run(["cargo", "build", *flag, *binaries])
    name = "cargo build (ladder binaries)"
    if code != 0:
        return Step(name, f"exit {code}: {last_nonempty(output)}", output=output)
    return Step(name, f"finished ({profile})", ok=True, output=output)


def binding_audit():
    code, output = run(
        [
            sys.executable,
            str(CRATE / "validation" / "audit_shader_binding_access.py"),
            "--check",
            "--json",
            str(SCRATCH / "binding_access.json"),
        ]
    )
    detail = last_nonempty(output)
    return Step("binding-access audit", detail, ok=code == 0, output=output)


SHADER_TOOLS = ("glslangValidator", "glslc", "spirv-val")
TOOLS_ROOT = CRATE.parent / ".tools"
INSTALL_SCRIPT = CRATE / "validation" / "install_shader_toolchain.sh"
TOOLCHAIN_HINT = (
    "install the pinned releases with `bash validation/install_shader_toolchain.sh` "
    "(Windows x86-64; on Linux take glslangValidator 16.5.0, glslc 2025.3 and "
    "SPIRV-Tools from the distribution or the Vulkan SDK), then re-run -- see "
    "VENDOR_TUNING.md, *Toolchain pin*"
)


def toolchain_directories(root=TOOLS_ROOT):
    """Where the installer puts the pinned binaries, in `PATH` order."""
    directories = sorted(path for path in root.glob("glslang-*/bin") if path.is_dir())
    shaderc = root / "env" / "Library" / "bin"
    if shaderc.is_dir():
        directories.append(shaderc)
    return directories


def ensure_shader_toolchain(
    *,
    lookup=shutil.which,
    directories=toolchain_directories,
    install=None,
    environment=None,
):
    """Make the pinned shader toolchain available and say what it took.

    Returns `(missing, detail)`. The repository-local `.tools/` install goes on
    `PATH` ahead of whatever the machine already has -- the pinned glslang has to
    resolve, not the one a shaderc package depends on. If the tools are still
    missing, `install()` is called once (the ladder passes a runner for
    `validation/install_shader_toolchain.sh`) and the tools are looked for again.
    Every dependency is injected so all of those paths are testable without a
    compiler, a download, or a GPU.
    """
    environment = os.environ if environment is None else environment
    actions: list[str] = []
    added: list[str] = []
    missing = [name for name in SHADER_TOOLS if lookup(name) is None]
    while missing:
        fresh = [str(path) for path in directories() if str(path) not in added]
        if fresh:
            added.extend(fresh)
            environment["PATH"] = os.pathsep.join([*fresh, environment.get("PATH", "")])
            actions.append(f"prepended {', '.join(fresh)} to PATH")
        elif install is not None:
            code, message = install()
            install = None
            if code != 0:
                actions.append(f"installer exit {code}: {message}")
                break
            actions.append("installed the pinned releases into .tools/")
        else:
            break
        missing = [name for name in SHADER_TOOLS if lookup(name) is None]
    if missing:
        attempts = "; ".join(actions)
        return missing, (
            f"missing {', '.join(missing)}"
            + (f" after {attempts}" if attempts else "")
            + f"; {TOOLCHAIN_HINT}"
        )
    return [], "; ".join(actions) if actions else "the pinned releases resolve on PATH"


def install_shader_toolchain():
    """Run the repository's no-admin installer the way a developer would."""
    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT)], cwd=CRATE, text=True, capture_output=True, check=False
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()[-2000:]


def shader_toolchain(install):
    missing, detail = ensure_shader_toolchain(
        install=install_shader_toolchain if install else None
    )
    if missing:
        return Step("shader toolchain", detail, blocked=True)
    reported = []
    for name in SHADER_TOOLS:
        path = shutil.which(name)
        tool = probe(path) if path else None
        reported.append(
            f"{name} {tool.version}" if tool and tool.version else f"{name} (no release reported)"
        )
    return Step("shader toolchain", f"{detail}: {'; '.join(reported)}", ok=True)


def topology_decisions(profile):
    code, output = run([
        sys.executable,
        CRATE / "validation" / "verify_topology_decisions.py",
        "--probe",
        binary("training_step_probe", profile),
        "--output",
        SCRATCH / "topology_decisions.json",
    ])
    return Step("topology decisions", last_nonempty(output), ok=code == 0, output=output)


def shader_provenance():
    report = SCRATCH / "shader_provenance.json"
    code, output = run(
        [
            sys.executable,
            str(CRATE / "validation" / "verify_training_shaders.py"),
            "--output",
            str(report),
        ]
    )
    if code == 0:
        try:
            variants = json.loads(report.read_text(encoding="utf-8"))["variants"]
            detail = f"{len(variants)} variant(s) match their sources and flags"
        except (OSError, KeyError, ValueError):
            detail = "rebuilt artifacts match their sources and flags"
        return Step("shader provenance", detail, ok=True, output=output)
    blocked = "must be on PATH" in output
    return Step(
        "shader provenance",
        f"exit {code}: {last_nonempty(output)}",
        blocked=blocked,
        output=output,
    )


def push_constant_layouts():
    code, output = run(
        [sys.executable, CRATE / "validation" / "check_push_constant_layouts.py"]
    )
    return Step(
        "push-constant layouts",
        last_nonempty(output),
        ok=code == 0,
        output=output,
    )


def oracle_parity():
    """Shaders against the read-only stable baseline, where one is available.

    Exit 2 is the checker's "oracle not present on this machine" skip, which is a
    blocked step rather than a pass: a machine without the baseline must not
    report parity it never checked.
    """
    code, output = run(
        [sys.executable, CRATE / "validation" / "check_oracle_parity.py"]
    )
    if code == 2:
        return Step("oracle parity", last_nonempty(output), blocked=True, output=output)
    return Step("oracle parity", last_nonempty(output), ok=code == 0, output=output)


def golden_harness(name, harness, golden, profile):
    executable = binary(harness, profile)
    if not executable.exists():
        return Step(name, f"{executable} is missing; do not pass --skip-build", blocked=True)
    if not golden.exists():
        return Step(
            name,
            f"{golden} is missing; record it with --write-golden before enforcing this step",
            blocked=True,
        )
    # Exercise both reduction comparison modes on every run. Comparing two
    # independently dispatched copies also detects a comparison accidentally
    # moved before the candidate dispatch, even without old artifacts in CI.
    baseline = [CRATE / "shaders"] if harness == "reduction_kernel_validation" else []
    code, output = run([executable, *baseline, "--golden", golden, "--golden-strict"])
    return Step(name, last_nonempty(output), ok=code == 0, output=output)


def expectation_problems(observed, expectation, device):
    """Compare one probe run against the expectation recorded for its device.

    `mean_loss` and `dispatch_count` are read from that device's entry alone: the
    vendor layer selects different kernels on different parts, so another device's
    numbers say nothing about this one and are not compared. `probe_args` and the
    barrier bound come from the fixture instead -- the bound is an upper bound, so
    an elision that removes barriers still passes. A missing device entry is a
    failure rather than a pass: the ladder is only green where it measured.
    """
    problems = []
    if expectation.get("probe_args") != PROBE_ARGS:
        problems.append(f"recorded probe arguments {expectation.get('probe_args')} do not match {PROBE_ARGS}")
    recorded = expectation.get("devices", {}).get(device, {})
    expected_dispatches = recorded.get("dispatch_count")
    if expected_dispatches is None:
        problems.append(f"no dispatch count recorded for {device}; run with --record-expectation on this device")
    elif observed["dispatch_count"] != expected_dispatches:
        problems.append(f"dispatch count {observed['dispatch_count']} != recorded {expected_dispatches}")
    barrier_max = expectation.get("shader_barrier_count_max")
    if barrier_max is not None and observed["shader_barrier_count"] > barrier_max:
        problems.append(
            f"barrier count {observed['shader_barrier_count']} exceeds the recorded upper bound {barrier_max}"
        )
    recorded_loss = recorded.get("mean_loss")
    if recorded_loss is None:
        problems.append(f"no expectation recorded for {device}; run with --record-expectation on this device")
    elif observed["mean_loss"] != recorded_loss:
        problems.append(f"loss {observed['mean_loss']} != recorded {recorded_loss}")
    return problems


def training_step(record, profile):
    executable = binary("training_step_probe", profile)
    if not executable.exists():
        return Step("training-step probe", f"{executable} is missing; do not pass --skip-build", blocked=True)
    fixture = SCRATCH / "fresh_fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    code, output = run([executable, "--model-dir", fixture, *PROBE_ARGS])
    if code != 0:
        return Step("training-step probe", f"exit {code}: {last_nonempty(output)}", output=output)
    try:
        observed = json.loads(output[output.index("{"):])
    except (ValueError, KeyError) as err:
        return Step("training-step probe", f"unparseable output: {err}", output=output)

    device = observed["device"]
    expectation = json.loads(EXPECTATION.read_text(encoding="utf-8")) if EXPECTATION.exists() else {}
    devices = expectation.get("devices", {})

    if record:
        # This is a merge, not a rewrite: the numbers go into this machine's entry
        # and every other device's entry keeps the values it was recorded with.
        # `shader_barrier_count_max` is a device-independent bound, so it only ever
        # widens -- re-recording on a device with fewer barriers must not tighten a
        # bound another device was passing against.
        devices[device] = {
            "mean_loss": observed["mean_loss"],
            "dispatch_count": observed["dispatch_count"],
        }
        recorded = {key: value for key, value in expectation.items() if key != "dispatch_count"}
        recorded["note"] = EXPECTATION_NOTE
        recorded["probe_args"] = PROBE_ARGS
        recorded["shader_barrier_count_max"] = max(
            observed["shader_barrier_count"], expectation.get("shader_barrier_count_max", 0)
        )
        recorded["devices"] = devices
        EXPECTATION.parent.mkdir(parents=True, exist_ok=True)
        EXPECTATION.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
        return Step(
            "training-step probe",
            f"recorded for {device}: {observed['dispatch_count']} dispatches, "
            f"{observed['shader_barrier_count']} barriers, loss {observed['mean_loss']}",
            ok=True,
            output=output,
        )

    problems = expectation_problems(observed, expectation, device)
    barrier_max = expectation.get("shader_barrier_count_max")
    detail = (
        f"{observed['dispatch_count']} dispatches, {observed['shader_barrier_count']} barriers "
        f"(upper bound {barrier_max}), loss {observed['mean_loss']} on {device}"
    )
    if problems:
        return Step("training-step probe", "; ".join(problems), output=output)
    return Step("training-step probe", detail, ok=True, output=output)


def training_state(profile):
    code, output = run([
        sys.executable, CRATE / "validation" / "verify_training_state.py",
        "--probe", binary("training_step_probe", profile),
        "--output", SCRATCH / "training_state.json",
    ])
    return Step("training-state parity", last_nonempty(output), ok=code == 0, output=output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", action="store_true", help="use release binaries")
    parser.add_argument("--skip-build", action="store_true", help="do not run cargo build")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only the steps whose name contains this text (repeatable)",
    )
    parser.add_argument(
        "--record-expectation",
        action="store_true",
        help="record the probe's instance counts and loss for the device in this machine",
    )
    parser.add_argument("--receipt", type=Path, help="write a JSON summary of this run here")
    parser.add_argument(
        "--no-toolchain-install",
        action="store_true",
        help="report what to install for the shader toolchain instead of running its installer",
    )
    args = parser.parse_args(argv)

    profile = "release" if args.release else "debug"
    SCRATCH.mkdir(parents=True, exist_ok=True)

    ladder = [
        ("cargo build", lambda: build(profile, args.skip_build)),
        ("shader toolchain", lambda: shader_toolchain(not args.no_toolchain_install)),
        ("binding-access audit", binding_audit),
        ("shader provenance", shader_provenance),
        ("push-constant layouts", push_constant_layouts),
        ("oracle parity", oracle_parity),
        *[
            (name, lambda name=name, harness=harness, golden=golden: golden_harness(name, harness, golden, profile))
            for name, harness, golden in HARNESSES
        ],
        ("training-step probe", lambda: training_step(args.record_expectation, profile)),
        ("topology decisions", lambda: topology_decisions(profile)),
        ("training-state parity", lambda: training_state(profile)),
    ]

    selected = ladder
    if args.only:
        selected = [entry for entry in ladder if any(needle in entry[0] for needle in args.only)]
        if not selected:
            available = ", ".join(name for name, _ in ladder)
            print(f"--only {args.only} matched no step; available: {available}", file=sys.stderr)
            return 2

    steps = []
    for name, check in selected:
        step = check()
        if step is None:
            continue
        steps.append(step)

    width = max(len(step.name) for step in steps)
    print()
    for step in steps:
        print(f"{step.status:8} {step.name:<{width}}  {step.detail}")
    failures = [step for step in steps if not step.ok]
    print()
    print(f"{len(steps) - len(failures)}/{len(steps)} step(s) passed")

    for step in failures:
        body = step.output.strip()
        print(f"--- output of {step.name} ({step.status}) ---")
        print(body[-4000:] if body else "(no output)")

    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(
                [
                    {"step": step.name, "status": step.status, "detail": step.detail}
                    for step in steps
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Record one device's training-state trajectory into the training-state golden.

The golden is device-keyed: each case carries one trajectory per measured device,
and `verify_training_state.py` refuses a device it has no trajectory for. This
recorder adds one device's trajectory to every case of the existing golden and
never rewrites a trajectory that is already there, so recording on a new machine
cannot overwrite another machine's evidence.

The trajectory is recorded under the policy the golden names in its own `env` block
(`HIERARCHOS_VULKAN_TRAINING_PRECISION`), which is what makes the FP16-storage
golden reproducible: the same run under the FP32 policy dispatches a different
LN1/low-rank family and writes a different trajectory.

It runs the probe exactly the way the verifier does -- same cases, same per-case
arguments, same `--state-output` shape -- and writes nothing unless every case ran
and every case reported the same device.

Usage:
  python validation/record_training_state.py
  python validation/record_training_state.py --probe target/debug/training_step_probe.exe
  python validation/record_training_state.py --device "Intel(R) HD Graphics 520"
  python validation/record_training_state.py --golden validation/kernel_golden/training_state_fp16.json
  python validation/record_training_state.py --replace   # re-record this device

`--device` pins the expected name rather than choosing one: the trajectories are
keyed by what the probe reports, so a name that disagrees is an error instead of a
trajectory filed under a machine that does not exist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

CRATE = Path(__file__).resolve().parents[1]
GOLDEN = CRATE / "validation" / "kernel_golden" / "training_state.json"


def record(probe, golden=GOLDEN, device=None, replace=False):
    recorded = json.loads(Path(golden).read_text(encoding="utf-8"))
    if recorded.get("schema") != 1 or not recorded.get("cases"):
        raise RuntimeError("unsupported or empty training-state golden file")
    scratch = CRATE / "target" / "record_training_state"
    scratch.mkdir(parents=True, exist_ok=True)

    # What has to survive the merge untouched: every device already recorded, the
    # case list, and the baseline block.
    before = [(case["name"], case["args"], dict(case["devices"])) for case in recorded["cases"]]
    before_scaffold = {key: value for key, value in recorded.items() if key != "cases"}

    observed_device = None
    trajectories = []
    with tempfile.TemporaryDirectory(prefix="record-", dir=scratch) as temp:
        temp = Path(temp)
        for index, case in enumerate(recorded["cases"]):
            snapshot = temp / f"state_{index}.json"
            command = [str(probe), "--model-dir", str(temp / f"fixture_{index}"),
                       *case["args"], "--state-output", str(snapshot)]
            # The golden's own `env` is the training numerical policy its trajectories
            # were recorded under, so the recorder and the verifier read the policy from
            # the same place and cannot disagree about which kernels produced the
            # baseline.
            environment = {**os.environ, **recorded.get("env", {})}
            result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True,
                                    env=environment)
            if result.returncode:
                raise RuntimeError(f"{case['name']} probe exit {result.returncode}: {result.stderr[-3000:]}")
            output = json.loads(result.stdout[result.stdout.index("{"):])
            case_device = output["device"]
            if observed_device is None:
                observed_device = case_device
            elif case_device != observed_device:
                raise RuntimeError(
                    f"{case['name']} ran on {case_device!r} while the earlier cases ran on "
                    f"{observed_device!r}; one golden entry cannot mix devices"
                )
            trajectory = json.loads(snapshot.read_text(encoding="utf-8"))
            if not trajectory:
                raise RuntimeError(f"{case['name']} wrote an empty trajectory")
            if case_device in case["devices"] and not replace:
                raise RuntimeError(
                    f"{case['name']} already carries a {case_device!r} trajectory; "
                    f"re-record it with --replace only if you mean to replace that evidence"
                )
            trajectories.append(trajectory)

    if device is not None and device != observed_device:
        raise RuntimeError(
            f"the probe reports device {observed_device!r}, not the pinned {device!r}"
        )
    for case, trajectory in zip(recorded["cases"], trajectories):
        case["devices"][observed_device] = trajectory

    # The merge has to be additive, and this is enforced before anything is
    # written, because this file holds other machines' evidence. Every device that
    # was already recorded must come out of the merge untouched, the case list and
    # the baseline block must be unchanged, and where the device being recorded is
    # new the whole document minus that key must reproduce the file that was read
    # byte for byte.
    if {key: value for key, value in recorded.items() if key != "cases"} != before_scaffold:
        raise RuntimeError("refusing to write: the merge would change the golden's non-case blocks")
    merged_devices = [(case["name"], case["args"], case["devices"]) for case in recorded["cases"]]
    for (name, args, devices), (was_name, was_args, was_devices) in zip(merged_devices, before):
        if (name, args) != (was_name, was_args):
            raise RuntimeError(f"refusing to write: case {was_name!r} would be rewritten")
        for device, trajectory in was_devices.items():
            if devices.get(device) != trajectory:
                raise RuntimeError(
                    f"refusing to write: the {device!r} trajectory in {name!r} would change"
                )
    carried_over = any(observed_device in was_devices for _, _, was_devices in before)
    if not carried_over:
        whole = json.loads(json.dumps(recorded, indent=2) + "\n")
        for case in whole["cases"]:
            case["devices"].pop(observed_device, None)
        if json.dumps(whole, indent=2) + "\n" != Path(golden).read_text(encoding="utf-8").replace("\r\n", "\n"):
            raise RuntimeError(
                "refusing to write: adding this device would change more of the golden "
                "than its own trajectories"
            )
    Path(golden).write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    return observed_device, [
        {"case": case["name"], "steps": len(trajectory),
         "tensor_comparisons": sum(len(step["tensors"]) for step in trajectory)}
        for case, trajectory in zip(recorded["cases"], trajectories)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=CRATE / "target" / "debug" /
                        ("training_step_probe.exe" if os.name == "nt" else "training_step_probe"))
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--device")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    device, rows = record(args.probe.resolve(), args.golden, args.device, args.replace)
    steps = sum(row["steps"] for row in rows)
    digests = sum(row["tensor_comparisons"] for row in rows)
    for row in rows:
        print(f"recorded {row['case']}: {row['steps']} steps, {row['tensor_comparisons']} tensors")
    print(f"recorded {device!r}: {steps} training steps, {digests} exact tensor digests")
    print("now verify: python validation/verify_training_state.py --probe "
          f"{args.probe}")


if __name__ == "__main__":
    main()

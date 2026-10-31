"""Frozen joint 40-layer / 384-expert / top-6 CPU PyTorch qualification.

The released compression, KV/index sharing and Engram schedules run together.
Hidden/intermediate widths are 32, attention heads are 64-wide, and vocabulary,
context and Engram tables remain reduced for the local device. The fixed seed,
initialization, six-step prefill/decode/reset schedule and CPU arithmetic are
declared before native measurement. Every logit must satisfy <2e-7, rtol=0.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import validate_deepseek_v41_model as model
import validate_deepseek_v41_structure_model as structure
import validate_deepseek_v41_wide_model as wide

FIXTURE = model.FIXTURE.parent / "deepseek_v41_joint_model.json"
SPEC = dict(seed=4141111, layers=40, experts=384, engram=True, hidden=32, inter=32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path", type=Path, required=True)
    p.add_argument("--kernel-reference-path", type=Path, required=True)
    p.add_argument("--write-fixture", action="store_true")
    p.add_argument("--write-checkpoints", action="store_true",
                   help="rebuild local shards only after verifying frozen hashes")
    p.add_argument("--native-output", type=Path,
                   help="compare native logits and stages exported with HIERARCHOS_V41_JOINT_TRACE=1")
    p.add_argument("--trace-output", type=Path)
    p.add_argument("--observation-output", type=Path)
    args = p.parse_args()
    if args.write_fixture and FIXTURE.exists():
        raise FileExistsError("joint fixture already frozen; repair native execution")
    if args.write_fixture and args.write_checkpoints:
        raise ValueError("choose initial freeze or hash-verified reconstruction")
    frozen = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
    if frozen is None and not args.write_fixture:
        raise FileNotFoundError("initial CPU freeze required before native qualification")
    print(f"Generating fixed joint CPU graph {SPEC}", flush=True)
    observer = structure.StructureObserver()
    expected_files = frozen["cases"][0]["checkpoint"]["files"] if frozen else None
    case = structure.compact(
        structure.make_case(args.reference_path, args.kernel_reference_path, SPEC, observer),
        args.write_fixture or args.write_checkpoints, expected_files,
        prefix="deepseek_v41_joint")
    for step, trace in zip(case["steps"], observer.steps):
        step["routing"] = trace["routing"]
    result = dict(torch_version=torch.__version__, torch_revision=torch.version.git_version,
                  cases=[case])
    if args.observation_output:
        args.observation_output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    if args.trace_output:
        args.trace_output.write_text(json.dumps(dict(cases=[dict(seed=SPEC["seed"], steps=observer.steps)]),
                                               separators=(",", ":"), allow_nan=False))
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    elif result != frozen:
        raise AssertionError("joint fixture differs from the pinned CPU reference")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == 1
        actual = native["cases"][0]
        assert actual["seed"] == SPEC["seed"]
        assert actual["checkpoint_files"] == case["checkpoint"]["files"]
        model.compare_native(case, actual, fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        assert len(actual["steps"]) == len(case["steps"])
        for step, expected in zip(actual["steps"], case["steps"]):
            assert step["routing"] == expected["routing"], "joint routing bits differ"
        report = wide.compare_diagnostics(case, actual, dict(steps=observer.steps))
        assert report["trace_values"] == report["bit_exact_values"], report
        print(json.dumps(report), flush=True)
    print(f"PASS joint CPU oracle: {sum(len(s['logits']) for s in case['steps'])} complete logits; "
          "full released widths/tables/context remain unqualified", flush=True)


if __name__ == "__main__":
    main()

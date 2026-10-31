#!/usr/bin/env python3
"""Compare this tree's training trajectory against a stable checkout, bit for bit.

The optimization work is only safe if the shipped graph' train step still produces
the values the pre-optimization tree produced. This script turns that claim into a
command: it builds `training_step_probe` in both trees, runs every golden geometry
against one shared model package per geometry, and compares the full-state
snapshots the probe writes with `--state-output`.

The comparison is on raw FP32 bit patterns, exactly as
`verify_training_state.py` compares a run against its golden: every trainable
parameter, both Adam moments, the per-slot optimizer clocks, token losses, the
final recurrent states, the initial-state adjoints and the context carriers, after
every update. One differing digest is a failure. Timings, dispatch counts and
losses are reported beside it, but they are not part of the gate: a legitimate
scheduling change may move the dispatch count, and none of them may move a value.

Neither tree is trusted: the *stable* checkout is the oracle, and its probe is
built from its own sources. The model package is initialized once per geometry by
the newer probe and passed to both legs, so both read byte-identical initial
weights -- the probe only ever reads the package it is handed, which is what makes
one package per geometry enough.

Usage:
    python hierarchos-vulkan/validation/check_stable_probe_parity.py \\
        --stable ../Hierarchos-Native-stable/Hierarchos-Native
    python hierarchos-vulkan/validation/check_stable_probe_parity.py \\
        --stable <checkout> --profile debug --case odd_multi_sequence
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_training_state as state  # noqa: E402  (the golden's cases and comparison)
from check_stable_logits import identity

CRATE = Path(__file__).resolve().parents[1]
BINARY = "training_step_probe"
CASES = state.load_golden(state.GOLDEN_DIR / "training_state.json")["cases"]
PROBE_MASK_ENV = "HIERARCHOS_VALIDATION_PROBE_MASK"
PROBE_GRADIENT_ENV = "HIERARCHOS_VALIDATION_PROBE_GRADIENTS"
PROBE_RESUME_ENV = "HIERARCHOS_VALIDATION_PROBE_RESUME"
EXTENDED_CASES = [
    {"name": "long_sequence128", "args": ["--batch", "1", "--tokens", "128", "--hidden", "32",
        "--context-dim", "32", "--warmup", "0", "--steps", "2"]},
    {"name": "masked_odd_accumulation3", "args": ["--batch", "3", "--tokens", "5", "--sequences", "3",
        "--hidden", "32", "--context-dim", "32", "--warmup", "0", "--steps", "3"],
        "env": {PROBE_MASK_ENV: "1"}},
    {"name": "raw_gradient_accumulation", "args": ["--batch", "3", "--tokens", "5", "--sequences", "3",
        "--hidden", "32", "--context-dim", "32", "--warmup", "0", "--steps", "3"],
        "env": {PROBE_MASK_ENV: "1", PROBE_GRADIENT_ENV: "1"}},
    {"name": "checkpoint_resume", "args": ["--batch", "2", "--tokens", "5",
        "--hidden", "32", "--context-dim", "32", "--warmup", "0", "--steps", "3"],
        "env": {PROBE_RESUME_ENV: "1"}},
    *({"name": f"batch{batch}_tokens16", "args": ["--batch", str(batch), "--tokens", "16",
        "--hidden", "128", "--context-dim", "128", "--warmup", "0", "--steps", "2"]}
      for batch in (1, 4, 8, 9, 16, 17, 32, 64)),
    {"name": "batch64_updates20", "args": ["--batch", "64", "--tokens", "8",
        "--hidden", "128", "--context-dim", "128", "--warmup", "0", "--steps", "20"]},
]


def extended_catalog() -> list[dict]:
    """Both storage policies compare the union, including rank-128 FP16 work."""
    cases = {case["name"]: {"name": case["name"], "args": case["args"]} for case in CASES}
    for case in state.load_golden(state.GOLDEN_DIR / "training_state_fp16.json")["cases"]:
        cases.setdefault(case["name"], {"name": case["name"], "args": case["args"]})
    cases.update((case["name"], case) for case in EXTENDED_CASES)
    cases["odd_multi_sequence"]["env_by_policy"] = {"fp16-storage-fp32-compute": {
        "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY": "one-row-rank128",
        "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG": "1"}}
    return list(cases.values())


class Mismatch(Exception):
    """One leg could not be measured, or the two legs disagree."""


def write_source(path: Path, source: str) -> None:
    """Keep Cargo's source timestamp when the generated observer is unchanged."""
    if not path.is_file() or path.read_text(encoding="utf-8") != source:
        path.write_text(source, encoding="utf-8")


def stable_crate(checkout: Path) -> Path:
    """Resolve a stable checkout's crate directory, refusing a path that has none."""
    checkout = checkout.resolve()
    crate = checkout / "hierarchos-vulkan"
    if not (crate / "Cargo.toml").exists():
        raise Mismatch(
            f"{checkout} is not a Hierarchos checkout: {crate / 'Cargo.toml'} does not exist. "
            "Pass the repository root (the directory that holds hierarchos-vulkan/)."
        )
    return crate


def build_probe(crate: Path, profile: str, inference_logits: bool = False,
                observer_source: Path | None = None) -> Path:
    """Build the probe in one tree and return the executable Cargo reported."""
    manifest = crate / "Cargo.toml"
    target_name = "stable-training-logit-observer" if inference_logits else BINARY
    if inference_logits:
        # Expose the same public checkpoint/inference boundary in BOTH libraries.
        # The stable sources stay read-only; only this isolated observer is new.
        observer = CRATE / "target" / "stable-logit-observers" / (
            "stable" if crate.resolve() != CRATE.resolve() else "candidate"
        )
        observer.mkdir(parents=True, exist_ok=True)
        source = (observer_source or crate / "src" / "bin" / "training_step_probe.rs").read_text(encoding="utf-8")
        # Read the clean oracle's observer source for BOTH libraries. Candidate
        # diagnostics must not accidentally make the two observers different.
        source = source.replace("    device: String,", "    device: String,\n    decision_device: String,")
        source = source.replace("        device: device.name().to_string(),", '''        device: device.name().to_string(),
        decision_device: format!("{} [pci={:04x}:{:04x};driver={:08x}]",
            device.name(), device.vendor_id(), device.device_id(), device.driver_version()),''')
        # Identical observation-only instrumentation in both trees. Registered
        # cases retain None; extended cases explicitly select deterministic masks.
        mask_seam = "    let mut snapshots = Vec::new();"
        mask_field = "pytorch_tbptt_token_mask: None,"
        if source.count(mask_seam) != 1 or source.count(mask_field) != 1:
            raise Mismatch("training probe mask seam changed; review the instrumentation")
        source = source.replace(mask_seam, '''    let validation_masks = (0..args.tokens)
        .map(|step| (0..args.batch).map(|row| if (step + row) % 3 == 0 { 0.0f32 } else { 1.0f32 })
            .collect::<Vec<_>>()).collect::<Vec<_>>();
    let validation_masked = std::env::var_os("HIERARCHOS_VALIDATION_PROBE_MASK").is_some();
    let validation_gradients = std::env::var_os("HIERARCHOS_VALIDATION_PROBE_GRADIENTS").is_some();
    let validation_resume = std::env::var_os("HIERARCHOS_VALIDATION_PROBE_RESUME").is_some();
''' + mask_seam)
        source = source.replace(mask_field,
            "pytorch_tbptt_token_mask: if validation_masked { Some(&validation_masks[step]) } else { None },")
        step_call = '''            let result = graph.train_raw_token_tape_microbatch_with_readback_policy(
                arena,
                &sequences,
                AdamWHyperParams::default(),
                HierarchosSequenceGradientNormalization::MeanByToken,
                args.readback,
            )?;'''
        if source.count(step_call) != 1:
            raise Mismatch("training probe update seam changed; review the gradient observer")
        source = source.replace(step_call, '''            let mut observed_gradients = BTreeMap::new();
            let result = if validation_gradients {
                let normalization_tokens = args.batch * args.tokens * args.sequences * 2;
                graph.train_raw_token_tape_microbatch_with_update_mode(
                    arena, &sequences, AdamWHyperParams::default(),
                    HierarchosSequenceGradientNormalization::MeanByToken,
                    hierarchos_vulkan::HierarchosTokenTapeUpdateMode::BeginAccumulation,
                    Some(normalization_tokens),
                )?;
                let result = graph.train_raw_token_tape_microbatch_with_update_mode(
                    arena, &sequences, AdamWHyperParams::default(),
                    HierarchosSequenceGradientNormalization::MeanByToken,
                    hierarchos_vulkan::HierarchosTokenTapeUpdateMode::Accumulate,
                    Some(normalization_tokens),
                )?;
                let gradients = graph.full_model_pending_gradient_snapshots()?;
                if !gradients.iter().any(|p| p.values.iter().any(|v| *v != 0.0)) {
                    bail!("gradient observer produced no nonzero raw parameter gradient");
                }
                for parameter in gradients {
                    observed_gradients.insert(format!("raw_gradient/{}", parameter.name),
                        vector_digest(&parameter.values)?);
                }
                observed_gradients.insert("gradient_normalization/token_count".to_owned(),
                    vector_digest(&[normalization_tokens as f32])?);
                graph.finish_full_model_accumulation(AdamWHyperParams::default())?;
                result
            } else {
                graph.train_raw_token_tape_microbatch_with_readback_policy(
                    arena, &sequences, AdamWHyperParams::default(),
                    HierarchosSequenceGradientNormalization::MeanByToken, args.readback,
                )?
            };''')
        source = source.replace("                let mut tensors = BTreeMap::new();",
            "                let mut tensors = observed_gradients;")
        marker = "                snapshots.push(StateSnapshot {"
        if source.count(marker) != 1:
            raise Mismatch("training probe observer seam changed; review the instrumentation")
        source = source.replace(marker, '''                let checkpoint = args.state_output.as_ref().unwrap()
                    .with_extension(format!("step-{}-package", snapshots.len()));
                graph.synchronize_ltm_alignment_controller_metadata()?;
                graph.export_training_checkpoint_package(&model_dir, &checkpoint)?;
                let inference = hierarchos_inference::HierarchosModel::load(&checkpoint)?;
                let mut inference_state = inference.new_state();
                for (position, token) in [1u32, 5, 9, 3, 7, 2].into_iter().enumerate() {
                    let logits = inference.step(token, &mut inference_state)?;
                    tensors.insert(format!("inference_logits/{position}"), vector_digest(&logits)?);
                }
                if validation_resume {
                    let mut resumed = HierarchosTrainingGraph::from_model_package_with_token_frontend(
                        device.clone(), &checkpoint, args.batch, 3, 2, args.batch, args.batch,
                    )?;
                    resumed.load_training_checkpoint_package_state(&checkpoint)?;
                    for parameter in resumed.full_model_parameter_snapshots()? {
                        if tensors.get(&format!("parameter/{}", parameter.name))
                            != Some(&vector_digest(&parameter.values)?) {
                            bail!("checkpoint parameter failed immediate reload: {}", parameter.name);
                        }
                    }
                    let restored_optimizer = resumed.full_model_optimizer_state()?;
                    if restored_optimizer.step != optimizer.step {
                        bail!("checkpoint optimizer clock failed immediate reload");
                    }
                    for slot in restored_optimizer.slots {
                        if optimizer_slot_steps.get(&slot.name) != Some(&slot.step)
                            || tensors.get(&format!("exp_avg/{}", slot.name)) != Some(&vector_digest(&slot.exp_avg)?)
                            || tensors.get(&format!("exp_avg_sq/{}", slot.name)) != Some(&vector_digest(&slot.exp_avg_sq)?) {
                            bail!("checkpoint optimizer slot failed immediate reload: {}", slot.name);
                        }
                    }
                    *graph = resumed;
                }
''' + marker)
        write_source(observer / "main.rs", source)
        manifest = observer / "Cargo.toml"
        write_source(manifest,
            '[package]\nname = "stable-logit-observer"\nversion = "0.0.0"\nedition = "2021"\n'
            f'[workspace]\n[[bin]]\nname = "{target_name}"\npath = "main.rs"\n'
            '[dependencies]\nanyhow = "1"\nserde = { version = "1", features = ["derive"] }\n'
            'serde_json = { version = "1", features = ["raw_value"] }\nsha2 = "0.10"\n'
            f'hierarchos-vulkan = {{ path = {json.dumps(crate.as_posix())} }}\n'
            f'hierarchos-inference = {{ path = {json.dumps((crate.parent / "hierarchos-inference").as_posix())} }}\n'
            '[profile.release]\nopt-level = 3\nlto = "thin"\ncodegen-units = 1\n',
        )
        if (crate / "Cargo.lock").is_file() and not (observer / "Cargo.lock").exists():
            shutil.copyfile(crate / "Cargo.lock", observer / "Cargo.lock")
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "--bin",
        target_name,
        "--message-format=json-render-diagnostics",
    ]
    if profile == "release":
        command.insert(2, "--release")
    if inference_logits:
        command += ["--target-dir", str(crate / "target"), "-j", "1"]
    completed = subprocess.run(command, cwd=crate.parent, capture_output=True, text=True)
    if completed.returncode:
        raise Mismatch(
            f"cargo could not build {BINARY} in {crate}: {completed.stderr.strip()[-2000:]}"
        )
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        if message["target"]["name"] == target_name:
            return Path(message["executable"])
    raise Mismatch(f"cargo reported no {BINARY} executable for {crate}")


def run_probe(
    binary: Path,
    crate: Path,
    model_dir: Path,
    case_args: list[str],
    snapshot: Path | None = None,
    precision: str = "fp32",
    case_environment: dict | None = None,
) -> tuple[dict, list[dict] | None]:
    """Run one probe leg and parse its report (and its snapshot, when asked for)."""
    command = [str(binary), "--model-dir", str(model_dir), *case_args]
    if snapshot is not None:
        command += ["--state-output", str(snapshot)]
    environment = dict(os.environ)
    for variable in (PROBE_MASK_ENV, PROBE_GRADIENT_ENV, PROBE_RESUME_ENV,
                     "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY",
                     "HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_AUTOTUNE_LOG"):
        environment.pop(variable, None)
    environment.update(case_environment or {})
    environment["HIERARCHOS_VULKAN_TRAINING_PRECISION"] = precision
    completed = subprocess.run(command, cwd=crate, capture_output=True, text=True, env=environment)
    if completed.returncode:
        raise Mismatch(
            f"{binary} exited {completed.returncode} on {case_args}: "
            f"{completed.stderr.strip()[-3000:] or completed.stdout.strip()[-3000:]}"
        )
    start = completed.stdout.find("{")
    if start < 0:
        raise Mismatch(f"{binary} printed no report on {case_args}")
    report = json.loads(completed.stdout[start:])
    arm = (case_environment or {}).get("HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY")
    if arm:
        report["forced_arm"] = {"name": arm, "exercised": state.arm_evidence(completed.stderr, arm)[0],
                                "unavailable": state.arm_evidence(completed.stderr, arm)[1]}
    states = None
    if snapshot is not None:
        states = json.loads(snapshot.read_text(encoding="utf-8"))
    return report, states


def initialize_package(binary: Path, crate: Path, package: Path, case_args: list[str]) -> None:
    """Let the newer probe create the fixture package both legs will read.

    The probe initializes a package when `--model-dir` does not already hold one
    and only ever reads it afterwards, so this run is the only writer. Its own
    report is discarded: the geometry it replays is not part of the comparison.
    """
    run_probe(binary, crate, package, case_args)
    if not (package / "hierarchos_config.json").exists():
        raise Mismatch(f"the probe did not initialize a model package at {package}")


def diff_snapshots(reference: list[dict], observed: list[dict]) -> list[str]:
    """Every difference between two probe snapshots, digest by digest."""
    return state.compare_snapshots(reference, observed)


def digests(states: list[dict]) -> int:
    return sum(len(step.get("tensors", {})) for step in states)


def reference_fan_in_environment(device: str | None, package: Path, args: list[str]) -> dict:
    """Reproduce the fixed reference association without changing its compute.

    The clean oracle's autotuner can choose two different FP32 programs. The
    existing goldens fix materialized association at batch32; all other shapes
    retain its original fused module. Only that oracle leg is constrained here.
    The shipping candidate must choose a compatible schedule on its own.
    """
    profiles = json.loads((CRATE / "validation" / "rwkv_fan_in_reference_profiles.json").read_text())
    if device != profiles["device"]:
        return {}
    from safetensors import safe_open
    with safe_open(str(package / "model.safetensors"), framework="np") as weights:
        shapes = [weights.get_slice(f"h_rnn.{name}").get_shape() for name in ("w1", "a1", "g1")]
    width = shapes[0][0]
    ranks = tuple(shape[1] for shape in shapes)
    batch = int(args[args.index("--batch") + 1]) if "--batch" in args else 2
    materialized = any(row["width"] == width
        and (row["w_rank"], row["a_rank"], row["g_rank"]) == ranks
        and row["max_batch"] == batch for row in profiles["materialized"])
    return {"HIERARCHOS_RWKV_LOW_RANK_FAN_IN_SCHEDULE": "split" if materialized else "fused-base",
            "HIERARCHOS_RWKV_BACKWARD_SEGMENT_CACHE_PATH": ""}


def compare_case(
    case: dict,
    *,
    stable_binary: Path,
    stable_crate_dir: Path,
    current_binary: Path,
    work: Path,
    precision: str = "fp32",
) -> dict:
    """Run one geometry through both trees and report exactly what differed."""
    args = list(case["args"])
    package = work / f"fixture_{case['name']}"
    if not package.exists():
        initialize_package(current_binary, CRATE, package, args)
    stable_snapshot = work / f"stable_{case['name']}.json"
    current_snapshot = work / f"current_{case['name']}.json"
    environment = {**case.get("env", {}), **case.get("env_by_policy", {}).get(precision, {})}
    current_report, current_states = run_probe(
        current_binary, CRATE, package, args, current_snapshot, precision, environment
    )
    reference_environment = {**environment, **reference_fan_in_environment(
        current_report.get("decision_device"), package, args)}
    stable_report, stable_states = run_probe(
        stable_binary, stable_crate_dir, package, args, stable_snapshot, precision, reference_environment
    )
    if stable_report["device"] != current_report["device"]:
        raise Mismatch(
            f"{case['name']}: the legs ran on different adapters "
            f"({stable_report['device']!r} and {current_report['device']!r})"
        )
    differences = diff_snapshots(stable_states, current_states)
    if environment.get("HIERARCHOS_RWKV_LOW_RANK_LN_FORWARD_TOPOLOGY"):
        if not current_report["forced_arm"]["exercised"]:
            differences.append("required FP16 rank-128 producer did not execute")
    if environment.get(PROBE_RESUME_ENV):
        control_environment = {k: v for k, v in environment.items() if k != PROBE_RESUME_ENV}
        for label, binary, crate, resumed in (("stable", stable_binary, stable_crate_dir, stable_states),
                                             ("candidate", current_binary, CRATE, current_states)):
            leg_environment = {**control_environment, **(reference_fan_in_environment(
                current_report.get("decision_device"), package, args) if label == "stable" else {})}
            _, uninterrupted = run_probe(binary, crate, package, args,
                work / f"{label}_{case['name']}_uninterrupted.json", precision, leg_environment)
            differences.extend(f"{label} resume continuation: {difference}"
                               for difference in diff_snapshots(uninterrupted, resumed))
    return {
        "case": case["name"],
        "precision": precision,
        "args": args,
        "environment": environment,
        "stable_reference_environment": reference_environment,
        "forced_arm": current_report.get("forced_arm"),
        "device": current_report["device"],
        "steps": len(current_states),
        "digests": min(digests(stable_states), digests(current_states)),
        "stable_dispatch_count": stable_report["dispatch_count"],
        "current_dispatch_count": current_report["dispatch_count"],
        "stable_mean_loss": stable_report["mean_loss"],
        "current_mean_loss": current_report["mean_loss"],
        "differences": differences,
        "pass": not differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stable",
        type=Path,
        required=True,
        help="the stable checkout to compare against (the directory holding hierarchos-vulkan/)",
    )
    parser.add_argument(
        "--profile",
        choices=("release", "debug"),
        default="release",
        help="build profile for both probes (default release)",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="golden geometry to compare (repeatable; default all of them)",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="write the comparison as JSON to this path as well as printing it",
    )
    parser.add_argument(
        "--keep-results",
        type=Path,
        help="keep the probe snapshots in this directory instead of a temporary one",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16-storage-fp32-compute", "both"),
                        default="fp32", help="compare each storage policy with the SAME stable policy")
    parser.add_argument("--with-inference-logits", action="store_true",
                        help="also export/reload after EVERY update and compare six recurrent logit vectors")
    parser.add_argument("--extended", action="store_true",
                        help="compare both policies across the union of golden shapes, route batches, long/masked tapes, raw gradients and checkpoint resume")
    args = parser.parse_args()
    if args.extended and not args.with_inference_logits:
        parser.error("--extended requires --with-inference-logits for the identical isolated observers")

    catalog = extended_catalog() if args.extended else CASES
    known = {case["name"] for case in catalog}
    selected = catalog if not args.cases else [case for case in catalog if case["name"] in args.cases]
    unknown = sorted(set(args.cases or ()) - known)
    if unknown:
        raise Mismatch(f"unknown case(s) {unknown}; the golden records {sorted(known)}")
    if not selected:
        raise Mismatch("no golden geometry selected, so nothing would be compared")

    stable = stable_crate(args.stable)
    baseline_identity = identity(stable.parent)
    if baseline_identity["status"]:
        raise Mismatch("the stable reference must be clean")
    candidate_identity = identity(CRATE.parent)
    observer_source = stable / "src" / "bin" / "training_step_probe.rs"
    stable_binary = build_probe(stable, args.profile, args.with_inference_logits, observer_source)
    current_binary = build_probe(CRATE, args.profile, args.with_inference_logits, observer_source)

    cleanup = None
    if args.keep_results is not None:
        work = args.keep_results.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="hierarchos-stable-parity-")
        work = Path(cleanup.name)
    try:
        policies = ("fp32", "fp16-storage-fp32-compute") if args.precision == "both" else (args.precision,)
        rows = []
        for precision in policies:
            policy_work = work / precision
            policy_work.mkdir(parents=True, exist_ok=True)
            cases = selected
            if precision != "fp32" and not args.cases and not args.extended:
                cases = state.load_golden(state.GOLDEN_DIR / "training_state_fp16.json")["cases"]
            for case in cases:
                print(f"Comparing {precision}/{case['name']}", flush=True)
                rows.append(
                    compare_case(
                        case,
                        stable_binary=stable_binary,
                        stable_crate_dir=stable,
                        current_binary=current_binary,
                        work=policy_work,
                        precision=precision,
                    )
                )
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    total = sum(row["digests"] for row in rows)
    if identity(stable.parent) != baseline_identity or identity(CRATE.parent) != candidate_identity:
        raise Mismatch("source tree changed during validation; rerun on a quiet checkout")
    failures = [row for row in rows if not row["pass"]]
    receipt = {
        "result": "fail" if failures else "pass",
        "stable_crate": str(stable),
        "stable": baseline_identity,
        "candidate": candidate_identity,
        "inference_logits": args.with_inference_logits,
        "profile": args.profile,
        "cases": rows,
        "digests_compared": total,
        "differences": sum(len(row["differences"]) for row in rows),
    }
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"{row['case']:<20} {row['device']}: {row['steps']} step(s), "
            f"{row['digests']} digests, "
            f"dispatch {row['stable_dispatch_count']} -> {row['current_dispatch_count']}"
        )
    if failures:
        for row in failures:
            for difference in row["differences"]:
                print(f"FAIL: {row['case']}: {difference}", file=sys.stderr)
        print(
            f"FAIL: {receipt['differences']} difference(s) between this tree and {stable} "
            f"across {total} tensor digests",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {total} exact tensor digests over {len(rows)} golden geometrie(s) match "
        f"{stable.parent.name} bit for bit"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Mismatch as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)

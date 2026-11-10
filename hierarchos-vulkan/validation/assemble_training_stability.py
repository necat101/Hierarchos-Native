#!/usr/bin/env python3
"""Reassemble the stability receipt from a completed producer work directory.

The numerical producer ran to completion but its final JSON write rejected a
nested pathlib.Path in the adverse-run evidence. This assembler rebuilds every
derived field from the shipped per-step reports, adapters and states, without
executing the GPU workload again.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import check_training_stability as c
import qualify_training_phase16 as q
import qualify_training_stability as p

CRATE = p.CRATE
WORK = CRATE / "target/stability-full"
EVIDENCE = WORK / "evidence"
RECEIPT = p.RECEIPT_DIR / "current-qualification.json"


def load(
    tag: str,
    arm_id: str,
    lr: float,
    steps: int,
    start: int,
    profile: bool,
    adapter: Path | None = None,
) -> dict:
    report_path = WORK / "reports" / f"{tag}.json"
    stderr_path = WORK / "logs" / f"{tag}.err.txt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    blocks = q.parse_profile(stderr_path.read_text(encoding="utf-8", errors="replace")) if profile else []
    return {
        "tag": tag,
        "arm": p.ARMS[arm_id],
        "lr": lr,
        "report": report,
        "report_path": report_path,
        "stdout_path": WORK / "logs" / f"{tag}.out.txt",
        "stderr_path": stderr_path,
        "adapter_dir": adapter,
        "save_state": None,
        "resume_state": None,
        "returns": 0,
        "process_ms": float(report.get("total_wall_ms", 0.0)),
        "setup_ms": 0.0,
        "expected_steps": steps,
        "start_step": start,
        "blocks": blocks,
    }


def checked(run: dict) -> dict:
    p.check_harness(run)
    return run


def assemble() -> dict:
    sweep_steps = 100
    conv_steps = 100
    sweep: dict[str, dict] = {}
    for lr in p.SWEEP_LRS:
        sweep[str(lr)] = {}
        for arm_id in ("A", "B"):
            run = checked(load(f"sweep-{lr:g}-{arm_id}", arm_id, lr, sweep_steps, 0, False))
            sweep[str(lr)][arm_id] = {
                "report": run,
                "metrics": p.loss_metrics([r["loss"] for r in run["report"]["records"]]),
            }
        a = sweep[str(lr)]["A"]["report"]["report"]["records"]
        b = sweep[str(lr)]["B"]["report"]["report"]["records"]
        assert [r["loss_bits"] for r in a] == [r["loss_bits"] for r in b]
        assert [r["param_digest"] for r in a] == [r["param_digest"] for r in b]
    stable_lrs = [
        float(lr) for lr in p.SWEEP_LRS
        if sweep[str(lr)]["A"]["metrics"]["stable"]
        and sweep[str(lr)]["B"]["metrics"]["stable"]
    ]
    selected_lr = min(
        stable_lrs,
        key=lambda lr: (sweep[str(lr)]["A"]["metrics"]["late_mean"], lr),
    )

    seeds: dict[str, dict] = {}
    for seed in (13, 21):
        run = checked(load(f"seed-{seed}", "A", selected_lr, conv_steps, 0, False))
        metrics = p.loss_metrics([r["loss"] for r in run["report"]["records"]])
        assert metrics["stable"]
        seeds[str(seed)] = {"report": run, "metrics": metrics}

    runs: dict[str, dict] = {}
    arms: dict[str, dict] = {}
    for arm_id in ("A", "B", "C", "D"):
        run = checked(load(f"arm-{arm_id}", arm_id, selected_lr, conv_steps, 0, True))
        metrics = p.loss_metrics([r["loss"] for r in run["report"]["records"]])
        assert metrics["stable"], f"arm {arm_id} unstable"
        runs[arm_id] = run
        arms[arm_id] = {
            "config": p.ARMS[arm_id],
            "summary": p.summary(run, q.check_run(run)),
            "metrics": metrics,
        }
    comparisons = {
        "A_vs_B": p.compare_runs(runs["A"], runs["B"],
                                 ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "C_vs_D": p.compare_runs(runs["C"], runs["D"],
                                 ["loss_bits", "param_digest", "update_norm", "update_max_abs"]),
        "B_vs_D": p.drift(runs["B"], runs["D"]),
        "A_vs_C": p.drift(runs["A"], runs["C"]),
    }
    for key in ("A_vs_B", "C_vs_D"):
        assert comparisons[key]["bitwise_equal"]
        assert comparisons[key]["final_parameter_digest_equal"]
    for key in ("B_vs_D", "A_vs_C"):
        assert comparisons[key]["first_step_within_policy"]

    resume: dict[str, dict] = {}
    resume_runs: dict[str, dict] = {}
    for arm_id in ("E", "F", "H"):
        total, split = 12, 8
        u = checked(load(f"resume-{arm_id}-u", arm_id, selected_lr, total, 0, False,
                         WORK / "adapters" / f"resume-{arm_id}-u"))
        r = checked(load(f"resume-{arm_id}-r", arm_id, selected_lr, total - split, split, True,
                         WORK / "adapters" / f"resume-{arm_id}-r"))
        r2 = checked(load(f"resume-{arm_id}-r-save", arm_id, selected_lr, total - split, split, False,
                          WORK / "adapters" / f"resume-{arm_id}-r-save"))
        u_state = WORK / "states" / arm_id / "u"
        restate = WORK / "states" / arm_id / "r"
        u_opt = {
            "adapter_model.safetensors": u["adapter_dir"] / "adapter_model.safetensors",
            "adapter_config.json": u["adapter_dir"] / "adapter_config.json",
            "hierarchos_adapter_optimizer.safetensors": u_state / "hierarchos_adapter_optimizer.safetensors",
        }
        r_opt = {
            "adapter_model.safetensors": r2["adapter_dir"] / "adapter_model.safetensors",
            "adapter_config.json": r2["adapter_dir"] / "adapter_config.json",
            "hierarchos_adapter_optimizer.safetensors": restate / "hierarchos_adapter_optimizer.safetensors",
        }
        u_hashes = {name: q.sha256_file(path) for name, path in u_opt.items()}
        r_hashes = {name: q.sha256_file(path) for name, path in r_opt.items()}
        u_bits = [x["loss_bits"] for x in u["report"]["records"][split:]]
        r_bits = [x["loss_bits"] for x in r["report"]["records"]]
        assert u_bits == r_bits, f"resume {arm_id} loss bits"
        assert u_hashes == r_hashes, f"resume {arm_id} checkpoint bytes"
        assert r["report"]["parameter_frozen"] is True
        resume[arm_id] = {
            "total_steps": total,
            "split_step": split,
            "checkpoint_sha256": {
                name: {"uninterrupted": u_hashes[name], "resumed": r_hashes[name]}
                for name in u_hashes
            },
            "checkpoint_bytes_equal": True,
            "loss_bits_equal": True,
            "loss_bits": {"uninterrupted_tail": u_bits, "resumed": r_bits},
            "resumed_optimizer_steps": [x["optimizer_step"] for x in r["report"]["records"]],
            "dispatch": q.check_run(r)["dispatch"],
            "parameter_frozen": True,
            "adapter_sha256": {
                "uninterrupted": u_hashes["adapter_model.safetensors"],
                "resumed": r_hashes["adapter_model.safetensors"],
            },
        }
        resume_runs[arm_id] = {"u": u, "r": r, "r2": r2, "u_state": u_state, "restate": restate}

    adverse: dict[str, dict] = {"seq96": {}}
    for arm_id in ("A", "B"):
        run = checked(load(f"adverse-seq96-{arm_id}", arm_id, selected_lr, 20, 0, True))
        adverse["seq96"][arm_id] = {
            "summary": p.summary(run, q.check_run(run)),
            "metrics": p.loss_metrics([r["loss"] for r in run["report"]["records"]]),
        }
    repeats = [checked(load(f"repeat-{index}", "A", selected_lr, 20, 0, False)) for index in (1, 2)]
    assert [x["loss_bits"] for x in repeats[0]["report"]["records"]] == [
        x["loss_bits"] for x in repeats[1]["report"]["records"]
    ]
    assert repeats[0]["report"]["parameter_end_digest"] == repeats[1]["report"]["parameter_end_digest"]
    adverse["repeat"] = {
        "loss_bits_equal": True,
        "parameter_digest_equal": True,
        "steps": 20,
        "sentinel_scratch": "unavailable: two-pass scratch is private and has no public poison hook",
    }
    adverse["seeds"] = seeds

    performance: dict[str, dict] = {}
    for arm_id in ("A", "B"):
        samples = {"scalar": [], "twopass": []}
        for path in sorted((WORK / "reports").glob(f"perf-{arm_id}-*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            mode = "twopass" if "twopass" in path.stem else "scalar"
            samples[mode].extend(x["step_ms"] for x in report["records"][3:])
        performance[arm_id] = {
            "steps": 20,
            "warmup": 3,
            "rounds": 2,
            "samples": samples,
            **q.latency_stats(samples["scalar"], samples["twopass"]),
        }

    if EVIDENCE.exists():
        shutil.rmtree(EVIDENCE)
    EVIDENCE.mkdir(parents=True)
    for path in sorted((WORK / "reports").glob("*.json")):
        shutil.copy2(path, EVIDENCE / path.name)
    for path in sorted((WORK / "logs").glob("*.err.txt")):
        shutil.copy2(path, EVIDENCE / path.name)
    for arm_id in ("E", "F", "H"):
        bundle = resume_runs[arm_id]
        for label, run in (("u", bundle["u"]), ("r", bundle["r2"])):
            shutil.copy2(run["adapter_dir"] / "adapter_model.safetensors",
                         EVIDENCE / f"adapter-{arm_id}-{label}.safetensors")
        for phase, state in (("u", bundle["u_state"]), ("r", bundle["restate"])):
            target = EVIDENCE / "states" / arm_id / phase
            target.mkdir(parents=True, exist_ok=True)
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "hierarchos_adapter_optimizer.safetensors"):
                shutil.copy2(state / name, target / name)

    source_files = [
        "src/transformer.rs",
        "src/vulkan.rs",
        "src/wmma_linear.rs",
        "src/wmma.rs",
        "src/transformer/peft_state.rs",
        "src/lib.rs",
        "src/bin/training-stability.rs",
        "src/bin/transformer_train.rs",
        "Cargo.toml",
        "shaders/transformer_attention_backward.spv",
        "shaders/transformer_attention_backward_query_rdna3.spv",
        "shaders/transformer_attention_backward_key_rdna3.spv",
        "shaders/transformer_attention_backward_query_rdna3.comp",
        "shaders/transformer_attention_backward_key_rdna3.comp",
        "validation/training_shader_manifest.json",
        "validation/qualify_training_stability.py",
        "validation/check_training_stability.py",
        "validation/test_training_stability.py",
        "validation/assemble_training_stability.py",
    ]
    bindings = {relative: q.sha256_file(CRATE / relative) for relative in source_files}
    evidence_files = {
        str(path.relative_to(EVIDENCE)).replace("\\", "/"): q.sha256_file(path)
        for path in sorted(EVIDENCE.rglob("*"))
        if path.is_file()
    }
    dirty = q.git("status", "--porcelain", "--untracked-files=no")
    receipt = {
        "schema": p.SCHEMA,
        "kind": p.KIND,
        "generated_at": p.now(),
        "source_revision": q.git("rev-parse", "HEAD"),
        "source_clean": not dirty,
        "evidence_source_revision": "fafa54a21492f53dac972499384b79604963719b",
        "phase16_receipt": {
            "path": str(p.PHASE16_RECEIPT.relative_to(CRATE)).replace("\\", "/"),
            "sha256": q.sha256_file(p.PHASE16_RECEIPT),
        },
        "toolchains": {
            "rustc": q.command_version(["rustc", "--version"]),
            "cargo": q.command_version(["cargo", "--version"]),
            "python": sys.version.split()[0],
        },
        "device": runs["A"]["report"]["device"],
        "fixture": {
            "model": str(p.MODEL.relative_to(CRATE)).replace("\\", "/"),
            "dataset": str(p.DATASET.relative_to(CRATE)).replace("\\", "/"),
            "model_config_sha256": q.sha256_file(p.MODEL / "config.json"),
            "model_weights_sha256": q.sha256_file(p.MODEL / "model.safetensors"),
            "dataset_sha256": q.sha256_file(p.DATASET),
            "parameters": p.EXPECTED_PARAMETERS,
        },
        "binaries": {
            "harness": q.sha256_file(p.BIN),
            "trainer": q.sha256_file(p.TRAINER),
        },
        "sweep": {
            "steps": sweep_steps,
            "lrs": [float(lr) for lr in p.SWEEP_LRS],
            "criterion": p.CRITERION,
            "selected_lr": selected_lr,
            "stable_lrs": stable_lrs,
            "runs": {
                lr: {
                    arm_id: {
                        "metrics": entry["metrics"],
                        "summary": p.summary(entry["report"], q.check_run(entry["report"])),
                    }
                    for arm_id, entry in arms_for_lr.items()
                }
                for lr, arms_for_lr in sweep.items()
            },
        },
        "seeds": {
            seed: {"metrics": entry["metrics"],
                   "summary": p.summary(entry["report"], q.check_run(entry["report"]))}
            for seed, entry in seeds.items()
        },
        "arms": arms,
        "comparisons": comparisons,
        "resume": resume,
        "adverse": adverse,
        "performance": performance,
        "limitations": [
            "full-parameter raw gradient tensors are not exposed; per-step gradient norms are recorded only for LoRA",
            "full-parameter exact checkpoint resume is not implemented and is refused",
            "two-pass scratch has no public sentinel/poison hook; sentinel contamination is not claimed",
            "mixed WMMA remains opt-in",
            "memory, scratch-byte overhead and checkpoint-export timing are not measured",
            "the sweep and convergence measure training loss, not held-out generalization",
        ],
        "source_files": bindings,
        "evidence": evidence_files,
    }
    work_receipt = WORK / "current-qualification.json"
    work_receipt.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    if p.RECEIPT_DIR.exists():
        shutil.rmtree(p.RECEIPT_DIR)
    p.RECEIPT_DIR.mkdir(parents=True)
    shutil.copytree(EVIDENCE, p.RECEIPT_DIR / "evidence")
    shutil.copy2(work_receipt, RECEIPT)
    print(f"installed={p.RECEIPT_DIR}")
    print(f"selected_lr={selected_lr:g} stable={stable_lrs}")
    print(f"evidence_files={len(evidence_files)}")
    return receipt


if __name__ == "__main__":
    assemble()

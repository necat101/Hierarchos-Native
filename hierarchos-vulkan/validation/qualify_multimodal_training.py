"""Freeze CPU projected-feature training, then run the same gate on Gen9/RDNA3.

This qualifies the embedding bridge and affine projector, not a modality tower.
Preparation never runs native code. Execution needs only the Python stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2e-7
LOSS_LIMIT = 2e-5  # Existing full-parameter training loss gate; logits stay strict.


def oracle_env(capability):
    """Replay the frozen CPU arithmetic on hosts with a different ISA."""
    width = {"AVX2": 8, "AVX512": 16}.get(capability.upper())
    if width is None:
        raise ValueError("unsupported frozen ATen vector policy")
    return {**os.environ, "HIERARCHOS_ATEN_VECTOR_WIDTH": str(width)}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(actual, expected):
    if not expected or len(actual) != len(expected):
        raise ValueError("empty or inconsistent comparison coverage")
    if not all(math.isfinite(x) for x in [*actual, *expected]):
        raise ValueError("non-finite value")
    worst = max(abs(a - b) for a, b in zip(actual, expected))
    return {"values": len(expected), "max_abs": worst, "pass": worst < LIMIT}


def read_f32_tensors(path):
    """Read reduced FP32 SafeTensors reference/export files without Torch."""
    data = Path(path).read_bytes()
    header_len, = struct.unpack_from("<Q", data)
    header = json.loads(data[8:8 + header_len])
    result = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if entry["dtype"] != "F32":
            continue
        start, end = entry["data_offsets"]
        count = math.prod(entry["shape"])
        if end - start != count * 4 or 8 + header_len + end > len(data):
            raise ValueError(f"invalid tensor range: {key}")
        result[key] = (entry["shape"], list(struct.unpack_from(
            f"<{count}f", data, 8 + header_len + start)))
    return result


def prepare(args):
    import torch
    from safetensors.torch import save_file
    import verify_hf_logits as reference

    torch.set_num_threads(1)
    torch.manual_seed(0x4D4D4739)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    models = reference.tiny_models(training_reference=True) + reference.production_width_models()
    from verify_falcon_h1 import tiny
    models.extend((name, tiny(norm=norm, before=before), False) for name, norm, before in [
        ("falcon_h1_gated_after", True, False), ("falcon_h1_gated_before", True, True),
        ("falcon_h1_silu", False, False)])
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    models.append(("qwen3_next", Qwen3NextForCausalLM(Qwen3NextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
        attention_dropout=0.0, use_cache=False, tie_word_embeddings=True,
        attn_implementation="eager", head_dim=8, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_conv_kernel_dim=3, num_experts=2, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        mlp_only_layers=[0, 1], layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0})), False))
    if args.families:
        missing = set(args.families) - {name for name, _, _ in models}
        if missing:
            raise ValueError(f"missing reference models: {sorted(missing)}")
        models = [entry for entry in models if entry[0] in args.families]
    manifest = {"schema": 1, "complete": False, "scope": "projected_features_full_parameter_training",
        "absolute_limit": LIMIT, "comparison": "strict_less_than", "rtol": 0,
        "loss_absolute_limit": LOSS_LIMIT, "cases": [], "deferred": [],
        "oracle": {"torch": torch.__version__, "torch_revision": torch.version.git_version,
            "cpu": torch.backends.cpu.get_cpu_capability(), "threads": torch.get_num_threads(),
            "transformers_revision": subprocess.check_output(["git", "-C",
                str(reference.TRANSFORMERS_SOURCE.parent), "rev-parse", "HEAD"], text=True).strip(),
            "transformers_status": subprocess.check_output(["git", "-C",
                str(reference.TRANSFORMERS_SOURCE.parent), "status", "--porcelain"], text=True).splitlines()}}
    source_dir = out / "oracle-source"
    source_dir.mkdir()
    for name in [Path(__file__).name, "verify_hf_logits.py", "kimi_k3_reference.py", "verify_falcon_h1.py"]:
        shutil.copyfile(Path(__file__).parent / name, source_dir / name)
    manifest["oracle"]["source_hashes"] = {p.relative_to(out).as_posix(): digest(p)
        for p in source_dir.iterdir()}
    for name, model, _ in models:
        if model.config.is_encoder_decoder:
            manifest["deferred"].append({"family": name, "reason": "seq2seq bridge integration pending"})
            continue
        # Match the existing deterministic full-parameter training oracle;
        # router-jitter training requires an explicit shared-RNG qualification.
        for module in model.modules():
            for field in ("jitter_noise", "moe_jitter_eps"):
                if hasattr(module, field):
                    setattr(module, field, 0.0)
        if hasattr(model.config, "router_jitter_noise"):
            model.config.router_jitter_noise = 0.0
        if name == "dbrx":
            model.config.ffn_config.moe_jitter_eps = 0.0
        for padded in (False, True):
            if padded and name in reference.KIMI_K3_ORACLE_FAMILIES:
                manifest["deferred"].append({"family": name, "case": "padded",
                    "reason": "existing Kimi CPU primitive oracle covers dense unpadded execution only"})
                continue
            # Reload the original model for each case; never carry updates from
            # another case into a CPU expectation or its initial checkpoint.
            import copy
            current = copy.deepcopy(model).float().train()
            for module in current.modules():
                if isinstance(module, torch.nn.Dropout) and module.p:
                    raise ValueError(f"{name}: nonzero dropout requires its own oracle")
            case = f"{name}/{'padded' if padded else 'unmasked'}"
            folder = out / case
            folder.mkdir(parents=True)
            reference.save_reference_checkpoint(name, current, folder / "model")
            embedding = current.get_input_embeddings()
            dim = embedding.weight.shape[1]
            feature_dim = 8
            generator = torch.Generator().manual_seed(0x4D4D + sum(name.encode()))
            features = (torch.randn(3, feature_dim, generator=generator) * .1).requires_grad_()
            projector = torch.nn.Linear(feature_dim, dim, bias=True)
            with torch.no_grad():
                projector.weight.copy_(torch.randn(dim, feature_dim, generator=generator) * .04)
                projector.bias.copy_(torch.randn(dim, generator=generator) * .01)
            ids = torch.tensor([[1, 5, 9, 3, 7, 2], [2, 7, 3, 9, 5, 1]])
            mask = torch.ones_like(ids)
            if padded:
                mask[0, -2:] = 0
                mask[1, 0] = 0
            positions = torch.tensor([8, 1, 2])  # Deliberately unsorted and across batches.
            targets = ids.roll(-1, dims=1).clone()
            weights = mask.float().clone()
            weights[:, -1] = 0
            weights[:, :-1] *= mask[:, 1:]
            for row in positions.tolist():
                batch, position = divmod(row, ids.shape[1])
                if position:
                    weights[batch, position - 1] = 0  # No image/audio target supervision.
            targets[weights == 0] = -100
            fixture = {"batch_size": 2, "seq_len": 6, "input_ids": ids.flatten().tolist(),
                "targets": [x if x >= 0 else 0xffffffff for x in targets.flatten().tolist()],
                "attention_mask": mask.float().flatten().tolist(), "loss_weights": weights.flatten().tolist(),
                "modality_rows": positions.tolist(), "features": features.detach().flatten().tolist(),
                "feature_dim": feature_dim, "projector_weight": projector.weight.detach().flatten().tolist(),
                "projector_bias": projector.bias.detach().tolist(), "steps": 2,
                "lr": .001, "beta1": .9, "beta2": .999, "eps": .001, "weight_decay": 0.0}
            write_json(folder / "input.json", fixture)
            optimizer = torch.optim.AdamW([*current.parameters(), *projector.parameters()],
                lr=fixture["lr"], betas=(fixture["beta1"], fixture["beta2"]), eps=fixture["eps"],
                weight_decay=0., foreach=False, fused=False)
            holder = {}

            def replace_rows(module, inputs, output):
                # Gemma3's lookup module includes its embedding scale. Our ABI
                # injects raw rows before that scale, so apply it here as well.
                replacement = holder["projected"] * getattr(module, "embed_scale", 1.0)
                return output.reshape(-1, dim).index_copy(0, positions, replacement).reshape_as(output)

            hook = embedding.register_forward_hook(replace_rows)
            cpu_steps = []
            for step in range(1, 3):
                optimizer.zero_grad(set_to_none=True)
                features.grad = None
                holder["projected"] = projector(features)
                holder["projected"].retain_grad()
                logits = current(input_ids=ids, attention_mask=mask, use_cache=False).logits
                losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1), reduction="none").reshape_as(weights)
                loss = (losses * weights).sum() / weights.sum()
                loss.backward()
                record = {"step": step, "loss": loss.item(), "logits": logits.detach().flatten().tolist(),
                    "projected": holder["projected"].detach().flatten().tolist(),
                    "modality_embedding_grad": holder["projected"].grad.flatten().tolist(),
                    "feature_grad": features.grad.flatten().tolist()}
                optimizer.step()
                record.update(projector_weight=projector.weight.detach().flatten().tolist(),
                    projector_bias=projector.bias.detach().flatten().tolist())
                cpu_steps.append(record)
                save_file({k: v.detach().cpu().contiguous().clone() for k, v in current.named_parameters()},
                    str(folder / f"cpu-step-{step}.safetensors"))
            with torch.no_grad():
                holder["projected"] = projector(features)
                final_logits = current(input_ids=ids, attention_mask=mask, use_cache=False).logits
            hook.remove()
            write_json(folder / "cpu.json", {"steps": cpu_steps,
                "final_logits": final_logits.flatten().tolist(),
                "aliases": getattr(current, "all_tied_weights_keys", {}),
                "visible": mask.bool().reshape(-1, 1).expand_as(final_logits.reshape(12, -1)).flatten().tolist()})
            files = {p.relative_to(out).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()}
            manifest["cases"].append({"family": name, "case": case, "files": files})
            write_json(out / "manifest.json", manifest)
            print(f"CPU {case}: two independent AdamW steps frozen", flush=True)
    if not manifest["cases"]:
        raise ValueError("no executable CPU cases")
    # Retain the upstream Python source actually imported by CPU preparation,
    # in addition to its Git revision. Replay of frozen values needs no import.
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            source = Path(name).resolve()
            if source.suffix == ".py" and source.is_relative_to(reference.TRANSFORMERS_SOURCE):
                destination = source_dir / source.relative_to(reference.TRANSFORMERS_SOURCE)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                manifest["oracle"]["source_hashes"][destination.relative_to(out).as_posix()] = digest(destination)
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)


def validate_device(report, target, device_id=None):
    if target == "gen9" and (report.get("vendor_id"), report.get("device_id")) != (0x8086, 0x1916):
        raise ValueError("expected the physical Intel HD Graphics 520")
    if target == "rdna3" and report.get("vendor_id") != 0x1002:
        raise ValueError("expected the physical AMD adapter")
    if target == "rdna3" and device_id is None:
        raise ValueError("RDNA3 qualification requires --device-id from the physical target")
    if device_id is not None and report.get("device_id") != device_id:
        raise ValueError("physical PCI device differs from --device-id")


def run(args):
    root = args.fixtures.resolve()
    out = args.output.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("CPU preparation is incomplete")
    if (manifest["absolute_limit"], manifest["comparison"], manifest["rtol"]) != (LIMIT, "strict_less_than", 0):
        raise ValueError("acceptance policy changed")
    environment = oracle_env(manifest["oracle"]["cpu"])
    out.mkdir(parents=True, exist_ok=False)
    cases = manifest["cases"]
    if args.families:
        missing = set(args.families) - {case["family"] for case in cases}
        if missing:
            raise ValueError(f"missing frozen families: {sorted(missing)}")
        cases = [case for case in cases if case["family"] in args.families]
    if not cases:
        raise ValueError("no qualification cases")
    results = []
    def persist():
        complete = len(results) == len(cases)
        write_json(out / "qualification.json", {"schema": 1, "target": args.target,
            "scope": manifest["scope"], "absolute_limit": LIMIT, "rtol": 0,
            "comparison": "strict_less_than", "loss_absolute_limit": LOSS_LIMIT,
            "expected_cases": len(cases), "completed_cases": len(results), "complete": complete,
            "fixtures_manifest_sha256": digest(root / "manifest.json"), "binary_sha256": digest(args.binary),
            "aten_vector_width": int(environment["HIERARCHOS_ATEN_VECTOR_WIDTH"]),
            "runner_sha256": digest(__file__), "results": results,
            "pass": complete and all(x["pass"] for x in results)})
    persist()
    for case in cases:
        for name, expected in {**manifest["oracle"]["source_hashes"], **case["files"]}.items():
            if digest(root / name) != expected:
                raise ValueError(f"frozen CPU input changed: {name}")
        source = root / case["case"]
        destination = out / case["case"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.binary.resolve()), "--model", str(source / "model"),
            "--fixture", str(source / "input.json"), "--output", str(destination)]
        process = subprocess.run(command, capture_output=True, text=True, env=environment)
        (destination.parent / f"{destination.name}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            results.append({"case": case["case"], "pass": False, "exit_code": process.returncode})
            persist()
            print(f"FAIL {case['case']}: native exit {process.returncode}", flush=True)
            continue
        report = json.loads((destination / "report.json").read_text())
        validate_device(report, args.target, args.device_id)
        cpu = json.loads((source / "cpu.json").read_text())
        metrics = []
        def check(label, actual, expected):
            result = compare(actual, expected)
            metrics.append({"metric": label, **result})
        def logits(label, actual, expected):
            # Match the existing CPU contract: observable query rows only.
            visible = cpu["visible"]
            if len(actual) != len(visible) or len(expected) != len(visible):
                raise ValueError("logit shape differs from visibility mask")
            if not all(math.isfinite(x) for x in [*actual, *expected]):
                raise ValueError("non-finite logit, including a padded row")
            check(label, [x for x, keep in zip(actual, visible) if keep],
                [x for x, keep in zip(expected, visible) if keep])
        if len(report["steps"]) != 2 or len(cpu["steps"]) != 2:
            raise ValueError("two optimizer steps required")
        for actual, expected in zip(report["steps"], cpu["steps"], strict=True):
            if actual["step"] != expected["step"]:
                raise ValueError("optimizer step differs")
            step = actual["step"]
            logits(f"step-{step}/logits", actual["logits"], expected["logits"])
            for key in ["projected", "modality_embedding_grad", "feature_grad", "projector_weight", "projector_bias"]:
                check(f"step-{step}/{key}", actual[key], expected[key])
            delta = abs(actual["loss"] - expected["loss"])
            metrics.append({"metric": f"step-{step}/loss", "max_abs": delta,
                "pass": math.isfinite(delta) and delta < LOSS_LIMIT})
            native = read_f32_tensors(destination / f"step-{step}" / "model.safetensors")
            for name, (shape, values) in read_f32_tensors(source / f"cpu-step-{step}.safetensors").items():
                key = name if name in native else cpu["aliases"].get(name, name)
                if key not in native or native[key][0] != shape:
                    raise ValueError(f"missing/reshaped native parameter {name}")
                check(f"step-{step}/parameter/{name}", native[key][1], values)
        logits("final_logits", report["final_logits"], cpu["final_logits"])
        result = {"case": case["case"], "device": report["device"],
            "vendor_id": report["vendor_id"], "device_id": report["device_id"],
            "pass": all(metric["pass"] for metric in metrics), "metrics": metrics}
        results.append(result)
        failures = [metric for metric in metrics if not metric["pass"]]
        print(f"{'PASS' if result['pass'] else 'FAIL'} {case['case']}: "
            f"{len(failures)} failing metrics" + (f"; worst {max(failures, key=lambda x: x['max_abs'])}" if failures else ""), flush=True)
        persist()
    if not all(x["pass"] for x in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--families", nargs="+")
    execute = sub.add_parser("run")
    execute.add_argument("--fixtures", type=Path, required=True)
    execute.add_argument("--binary", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--target", choices=["gen9", "rdna3"], required=True)
    execute.add_argument("--device-id", type=lambda value: int(value, 0))
    execute.add_argument("--families", nargs="+")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()

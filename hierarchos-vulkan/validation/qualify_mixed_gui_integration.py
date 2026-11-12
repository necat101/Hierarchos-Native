"""Exercise AMP-style WMMA/FP32 image training and its desktop CLI entry point.

This is an additional eligible-shape regression, not a replacement for frozen
Gen9/RDNA3 oracles or Phase XIX-Z. CPU references are computed independently in
PyTorch before GPU execution and recorded as diagnostics for this additional
eligible shape. The maintained frozen FP32 replay is a separate required gate.
Existing FP32 reload and mixed thresholds are retained.
Requires numpy, torch, safetensors, Pillow, and the local Transformers checkout.
Generated model/checkpoint files belong under target/, not in Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FP32_LIMIT = 2e-7
MIXED_MAX = 2e-2
MIXED_RMS = 3e-3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(folder, transformers, hidden_size=128, vocab_size=32):
    sys.path.insert(0, str(transformers / "src"))
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=hidden_size,
                        num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
                        head_dim=hidden_size // 8, max_position_embeddings=64, attention_dropout=0.,
                        tie_word_embeddings=False, use_cache=False,
                        attn_implementation="eager")
    text = LlamaForCausalLM(config).float().train()
    clip_config = CLIPVisionConfig(hidden_size=hidden_size, intermediate_size=hidden_size,
                                  num_hidden_layers=1, num_attention_heads=8,
                                  image_size=16, patch_size=4, hidden_act="quick_gelu",
                                  attention_dropout=0.)
    tower = CLIPVisionModel(clip_config).float().train()
    tower.set_attn_implementation("eager")
    projector = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.GELU(),
                                    torch.nn.Linear(hidden_size, hidden_size))
    with torch.no_grad():
        for parameter in projector.parameters():
            parameter.mul_(.15)
    package = folder / "model"
    (package / "image").mkdir(parents=True)
    text.save_pretrained(package / "text", safe_serialization=True)
    write(package / "config.json", dict(model_type="hierarchos_image_text", schema=1))
    write(package / "image/config.json", dict(
        model_type="hierarchos_clip_adapter", vision_config=clip_config.to_dict(),
        vision_feature_layer=-1, vision_feature_select_strategy="default",
        projector_hidden_act="gelu", multimodal_projector_bias=True, image_token_index=31))
    processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)
    processor.save_pretrained(package / "image")
    parameters = {"vision_tower.vision_model." + name: p for name, p in tower.named_parameters()}
    parameters.update({"multi_modal_projector.linear_1." + name: p for name, p in projector[0].named_parameters()})
    parameters.update({"multi_modal_projector.linear_2." + name: p for name, p in projector[2].named_parameters()})
    save_file({k: p.detach().contiguous().clone() for k, p in parameters.items()},
              str(package / "image/model.safetensors"))
    rgb = np.random.default_rng(20261112).integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    pixels = processor(images=[Image.fromarray(image) for image in rgb], return_tensors="pt")["pixel_values"].requires_grad_()
    ids = torch.tensor([[1] + [31] * 16 + [7, 3, 2, 5, 6, 4, 1],
                        [2] + [31] * 16 + [3, 7, 1, 6, 5, 2, 4]])
    targets = ids.roll(-1, dims=1)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    weights[:, 16:-1] = 1.
    fixture = dict(batch_size=2, seq_len=24, image_count=2,
                   input_ids=ids.flatten().tolist(), attention_mask=[1.] * 48,
                   targets=targets.flatten().tolist(), loss_weights=weights.flatten().tolist(),
                   rgb=rgb.flatten().tolist(), alternate_rgb=(255-rgb).flatten().tolist(),
                   steps=2, lr=.001, beta1=.9, beta2=.999, eps=.001, weight_decay=0.)
    write(folder / "input.json", fixture)
    rows = []
    for index in range(2):
        Image.fromarray(rgb[index]).save(folder / f"image-{index}.png")
        rows.append(dict(image=f"image-{index}.png", input_ids=ids[index].tolist(),
                         targets=targets[index].tolist(), loss_weights=weights[index].tolist()))
    (folder / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    positions = (ids == 31).flatten().nonzero().flatten()
    holder = {}

    def replace_rows(module, inputs, output):
        return output.reshape(-1, hidden_size).index_copy(0, positions, holder["projected"]).reshape_as(output)

    hook = text.get_input_embeddings().register_forward_hook(replace_rows)

    def forward():
        vision = tower(pixels, output_hidden_states=True).hidden_states[-1][:, 1:, :]
        holder["projected"] = projector(vision.reshape(-1, hidden_size))
        return text(ids, attention_mask=torch.ones_like(ids)).logits

    optimizer = torch.optim.AdamW([*text.parameters(), *parameters.values()],
                                 lr=.001, betas=(.9, .999), eps=.001,
                                 weight_decay=0., foreach=False, fused=False)
    steps = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        pixels.grad = None
        logits = forward()
        holder["projected"].retain_grad()
        losses = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.flatten(), reduction="none")
        loss = (losses * weights.flatten()).sum() / weights.sum()
        loss.backward()
        steps.append(dict(logits=logits.detach().flatten().tolist(), loss=loss.item(),
                          pixel_gradient=pixels.grad.flatten().tolist(),
                          projected_gradient=holder["projected"].grad.flatten().tolist()))
        optimizer.step()
    with torch.no_grad():
        final = forward().flatten().tolist()
    hook.remove()
    write(folder / "cpu.json", dict(steps=steps, final=final,
          torch=torch.__version__, numpy=np.__version__,
          transformers_revision=subprocess.check_output(["git", "-C", str(transformers), "rev-parse", "HEAD"], text=True).strip()))


def metric(reference, actual, max_limit, rms_limit=None):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    require(a.shape == b.shape and a.size > 0, "mismatched or empty numerical observation")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "non-finite numerical observation")
    delta = a-b
    result = dict(max_abs=float(abs(delta).max()), rms=float(np.sqrt(np.mean(delta*delta))))
    require(result["max_abs"] < max_limit, f"max_abs {result} exceeds strict {max_limit}")
    if rms_limit is not None:
        require(result["rms"] < rms_limit, f"RMS {result} exceeds strict {rms_limit}")
    return result


def cpu_diagnostic(reference, actual):
    result = metric(reference, actual, float("inf"))
    result.update(strict_limit=FP32_LIMIT, comparison="strict_less_than",
                  within_strict_limit=result["max_abs"] < FP32_LIMIT, qualified=False)
    return result


def run(args):
    from safetensors.numpy import load_file
    args.output.mkdir(parents=True, exist_ok=False)
    paths = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "hierarchos-vulkan/src", "hierarchos-vulkan/shaders",
                                    "hierarchos-native-cli/src", "hierarchos-gui/src", "hierarchos-inference/src",
                                    "hierarchos-vulkan/Cargo.*", "hierarchos-native-cli/Cargo.*",
                                    "hierarchos-gui/Cargo.*", "hierarchos-inference/Cargo.*"], text=True).splitlines()
    paths.append(str(Path(__file__).relative_to(ROOT)).replace("\\", "/"))
    source_hashes = {name: digest(ROOT / name) for name in paths}
    prepare(args.output, args.transformers)
    fixture_hashes = {path.relative_to(args.output).as_posix(): digest(path)
                      for path in args.output.rglob("*") if path.is_file()}
    cpu = json.loads((args.output / "cpu.json").read_text())
    reports, checks, commands = {}, {}, []
    binaries = {"probe": args.probe.resolve(), "cli": args.cli.resolve()}
    binary_hashes = {name: digest(path) for name, path in binaries.items()}
    for mode in ("off", "mixed"):
        env = {**os.environ, "HIERARCHOS_WMMA_MODE": mode}
        destination = args.output / mode
        command = [str(binaries["probe"]), "--model", str(args.output / "model"),
                   "--fixture", str(args.output / "input.json"), "--output", str(destination)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (args.output / f"{mode}.log").write_text(result.stdout+result.stderr, encoding="utf-8")
        commands.append(dict(command=command, mode=mode, returncode=result.returncode))
        require(result.returncode == 0, f"{mode} probe failed: {(result.stdout+result.stderr)[-1000:]}")
        report = json.loads((destination / "report.json").read_text())
        precision = report["precision"]
        require(precision["requested"] == mode, "wrong precision selection")
        require((precision["cooperative_dispatches"] > 0) == (mode == "mixed"), "incorrect physical dispatch selection")
        if mode == "mixed":
            require(all(precision[name] > 0 for name in ("forward", "weight_grad", "input_grad")), "missing WMMA training phase")
        observations = {}
        for key in ("logits", "vision_features", "projected_features"):
            observations["reload_" + key] = metric(report["final"][key], report["reload"][key], FP32_LIMIT)
        for index, step in enumerate(report["steps"]):
            require(step["training"]["text"]["step"] == index+1, "wrong optimizer step")
            for key in ("pixel_gradient", "projected_gradient"):
                require(any(abs(v) > 0 for v in step["training"][key]), "missing image gradient")
            if mode == "off":
                observations[f"cpu_step_{index+1}"] = cpu_diagnostic(cpu["steps"][index]["logits"], step["forward"]["logits"])
                for key in ("pixel_gradient", "projected_gradient"):
                    observations[f"cpu_{key}_{index+1}"] = cpu_diagnostic(cpu["steps"][index][key], step["training"][key])
        if mode == "off":
            observations["cpu_final"] = cpu_diagnostic(cpu["final"], report["final"]["logits"])
        for component in ("image", "text"):
            source = load_file(str(args.output / f"model/{component}/model.safetensors"))
            trained = load_file(str(destination / f"step-2/{component}/model.safetensors"))
            require(source.keys() == trained.keys(), "export changed parameter inventory")
            require(all(tensor.dtype.name == "float32" for tensor in trained.values()), "master checkpoint narrowed from FP32")
            changed = [name for name in source if (source[name] != trained[name]).any()]
            require(changed, f"no {component} parameters updated")
            if component == "image":
                require(any("vision_tower" in name for name in changed), "frozen image tower")
                require(any("projector" in name for name in changed), "frozen projector")
            observations[component + "_changed_parameters"] = len(changed)
        reports[mode], checks[mode] = report, observations
        # Exercise the same native command built by the desktop vision panel.
        # An opposing inherited policy verifies that the explicit flag wins.
        command = [str(binaries["cli"]), "vision-train", "--model-path", str(args.output / "model"),
                   "--train", str(args.output / "dataset.jsonl"), "--out-dir", str(args.output / f"cli-{mode}"),
                   "--wmma-mode", mode, "--batch-size", "2", "--max-length", "24", "--epochs", "2", "--lr", ".001",
                   "--eps", ".001", "--weight-decay", "0"]
        result = subprocess.run(command, env={**env, "HIERARCHOS_WMMA_MODE": "mixed" if mode == "off" else "off"}, capture_output=True, text=True)
        (args.output / f"cli-{mode}.log").write_text(result.stdout+result.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=result.returncode))
        require(result.returncode == 0, f"{mode} CLI failed: {(result.stdout+result.stderr)[-1000:]}")
        training = json.loads((args.output / f"cli-{mode}/training_report.json").read_text())
        require(training["steps"] == 2 and training["precision"]["requested"] == mode, "CLI precision/step mismatch")
        require((training["precision"]["cooperative_dispatches"] > 0) == (mode == "mixed"), "CLI dispatch mismatch")
        checks[mode]["cli_precision"] = training["precision"]
        for component in ("image", "text"):
            direct = load_file(str(destination / f"step-2/{component}/model.safetensors"))
            launched = load_file(str(args.output / f"cli-{mode}/model/{component}/model.safetensors"))
            require(direct.keys() == launched.keys(), "CLI/probe parameter inventory mismatch")
            checks[mode][f"cli_{component}_parameter_max_abs"] = max(
                metric(direct[name], launched[name], FP32_LIMIT)["max_abs"] for name in direct)
    for index in range(2):
        checks[f"mixed_step_{index+1}"] = metric(reports["off"]["steps"][index]["forward"]["logits"],
                                                 reports["mixed"]["steps"][index]["forward"]["logits"], MIXED_MAX, MIXED_RMS)
        for key in ("pixel_gradient", "projected_gradient"):
            checks[f"mixed_{key}_{index+1}"] = metric(reports["off"]["steps"][index]["training"][key],
                reports["mixed"]["steps"][index]["training"][key], MIXED_MAX, MIXED_RMS)
    text_runs = {}
    for family in ("full", "lora"):
        for mode in ("off", "mixed"):
            destination = args.output / f"text-{family}-{mode}"
            command = [str(binaries["cli"]), "transformer-train" if family == "full" else "transformer-finetune",
                       "--model-path", str(args.output / "model/text"),
                       "--train", str(args.output / "dataset.jsonl"), "--out-dir", str(destination),
                       "--amp" if mode == "mixed" else "--no-amp", "--batch-size", "2",
                       "--max-length", "24", "--epochs", "2", "--lr", ".001", "--eps", ".001",
                       "--weight-decay", "0", "--seed", "1234"]
            if family == "lora":
                command += ["--lora-rank", "8"]
            result = subprocess.run(command, env={**os.environ,
                "HIERARCHOS_WMMA_MODE": "off" if mode == "mixed" else "mixed"}, capture_output=True, text=True)
            log = result.stdout + result.stderr
            (args.output / f"text-{family}-{mode}.log").write_text(log, encoding="utf-8")
            commands.append(dict(command=command, returncode=result.returncode))
            require(result.returncode == 0, f"text {family}/{mode} failed: {log[-1000:]}")
            training = json.loads((destination / "training_report.json").read_text())
            steps = training["epochs"]
            require(training["steps"] == 2 and len(steps) == 2 and
                    [row["optimizer_step"] for row in steps] == [1, 2], "text optimizer steps missing")
            require(training["device"] == reports["mixed"]["device"], "text used a different device")
            require(all(math.isfinite(row["mean_loss"]) for row in steps), "non-finite text training loss")
            require(steps[1]["mean_loss"] < steps[0]["mean_loss"], "text loss did not reduce")
            precision = training["precision"]
            require(precision["requested"] == mode, "incorrect text precision selection")
            require((precision["cooperative_dispatches"] > 0) == (mode == "mixed"), "incorrect text dispatch selection")
            if mode == "mixed":
                require(precision["forward"] > 0 and precision["input_grad"] > 0, "missing text cooperative phases")
                require((precision["weight_grad"] > 0) == (family == "full"), "incorrect full/LoRA base-weight gradient dispatch")
                checks[f"text_{family}_dispatch"] = precision
            export = destination / ("model.safetensors" if family == "full" else "adapter_model.safetensors")
            require(export.is_file(), "text trained checkpoint was not exported")
            require(all(tensor.dtype.name == "float32" for tensor in load_file(str(export)).values()), "text checkpoint narrowed from FP32")
            text_runs[(family, mode)] = [row["mean_loss"] for row in steps]
        checks[f"text_{family}_loss_comparison"] = metric(text_runs[(family, "off")], text_runs[(family, "mixed")], MIXED_MAX, MIXED_RMS)
    require(all(digest(binaries[name]) == sha for name, sha in binary_hashes.items()), "binary changed during execution")
    require(all(digest(ROOT / name) == sha for name, sha in source_hashes.items()), "source changed during execution")
    require(all(digest(args.output / name) == sha for name, sha in fixture_hashes.items()), "fixture/oracle changed during execution")
    write(args.output / "qualification.json", dict(
        schema=1, status="PASS_SCOPED", scope="eligible shared CLIP + Llama full/LoRA two-step mixed/FP32 GUI command integration",
        source_revision=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        dirty=subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).splitlines(),
        source_sha256=source_hashes, binary_sha256=binary_hashes, fixture_sha256=fixture_hashes,
        evidence_sha256={path.relative_to(args.output).as_posix(): digest(path)
                         for path in args.output.rglob("*") if path.is_file()},
        device=reports["mixed"]["device"], vendor_id=reports["mixed"]["vendor_id"], device_id=reports["mixed"]["device_id"],
        checks=checks, commands=commands, cpu_oracle_sha256=digest(args.output / "cpu.json"),
        limitations=["Additional fixture; does not replace complete frozen multimodal qualification",
                     "Wider CPU oracle comparisons are diagnostic, with strict-threshold outcomes reported explicitly",
                     "No automatic loss scaling", "Native DeepSeek V4.1 WMMA is not implemented",
                     "No production-size readiness or performance claim", "No broad Phase XIX-Z qualification claim"]))
    print("MIXED GUI INTEGRATION: PASS_SCOPED (wide CPU oracle is diagnostic)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--transformers", type=Path, default=Path.home() / "transformers")
    options = parser.parse_args()
    options.output = options.output.resolve()
    run(options)

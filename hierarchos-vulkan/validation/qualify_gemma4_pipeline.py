"""Independent upstream Gemma4 oracle for raw RGB -> vision tower -> embedder -> logits.

CPU preparation never executes native code. Native replay is stdlib-only. The
first frozen geometry is deliberately the smallest real Gemma4 composite:
12x12 RGB, 2x2 patches, a 6x6 patch grid, 3x3 pooling to four soft tokens, two
vision layers, two text layers, and two complete composed AdamW updates.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from qualify_multimodal_training import compare, digest, read_f32_tensors, write_json, validate_device

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2e-7
CASES = [("gemma4_default", 12, 2, 3, 4), ("gemma4_rgb8", 12, 2, 3, 4)]


def oracle_env(capability):
    width = {"AVX2": 8, "AVX512": 16}.get(capability.upper())
    if width is None:
        raise ValueError("unsupported frozen ATen vector policy")
    return {**os.environ, "HIERARCHOS_ATEN_VECTOR_WIDTH": str(width)}


def patches_from_rgb(rgb, side, patch_size):
    import numpy as np
    image = rgb.astype(np.float32).reshape(side, side, 3).transpose(2, 0, 1)
    grid = side // patch_size
    patches = image.reshape(3, grid, patch_size, grid, patch_size)
    patches = patches.transpose(1, 3, 2, 4, 0).reshape(grid * grid, -1)
    return patches


def prepare(args):
    if args.cases and set(args.cases) - {case[0] for case in CASES}:
        raise ValueError("unknown requested pipeline case")
    import torch
    import numpy as np
    import PIL
    from PIL import Image
    from safetensors.torch import save_file
    checkout = Path(os.environ.get("HIERARCHOS_TRANSFORMERS_CHECKOUT", str(ROOT.parent / "transformers")))
    sys.path.insert(0, str(checkout / "src"))
    from transformers import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4TextConfig,
        Gemma4VisionConfig,
        Gemma4ImageProcessorPil,
    )
    torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "complete": False,
        "scope": "raw_rgb_gemma4_vision_embedder_full_parameter_training",
        "logit_absolute_limit": LIMIT,
        "parameter_absolute_limit": LIMIT,
        "comparison": "strict_less_than",
        "rtol": 0,
        "cases": [],
        "oracle": {
            "torch": torch.__version__,
            "torch_revision": torch.version.git_version,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "threads": 1,
            "cpu": torch.backends.cpu.get_cpu_capability(),
            "transformers_revision": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
            "transformers_status": subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).splitlines(),
        },
    }
    for case_index, (name, side, patch_size, pool, soft_tokens) in enumerate(CASES):
        if args.cases and name not in args.cases:
            continue
        torch.manual_seed(0xC11F800 + case_index)
        vision = Gemma4VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            hidden_act="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            patch_size=patch_size,
            pooling_kernel_size=pool,
            position_embedding_size=16,
            use_clipped_linears=False,
            standardize=False,
            # Explicit axial policy: default_rope_type alone is not serialized.
            rope_parameters={"rope_type": "axial", "rope_theta": 100.0},
        )
        text = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            layer_types=["full_attention", "full_attention"],
            sliding_window=0,
            hidden_activation="gelu_pytorch_tanh",
            attention_dropout=0.0,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            },
        )
        # Gemma4TextConfig materializes per-layer head geometry for full
        # attention. The native graph owns one uniform geometry here.
        text.per_layer_config = {}
        vision._attn_implementation = "eager"
        text._attn_implementation = "eager"
        config = Gemma4Config(
            text_config=text,
            vision_config=vision,
            audio_config=None,
            image_token_id=63,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        model = Gemma4ForConditionalGeneration(config).float().train()
        folder = out / name
        package = folder / "model"
        folder.mkdir()
        model.save_pretrained(package, safe_serialization=True, save_original_format=False)
        pixel_domain="rescaled_rgb8" if name=="gemma4_rgb8" else "raw_patch_values"
        processor=None
        if pixel_domain=="rescaled_rgb8":
            processor=Gemma4ImageProcessorPil(do_resize=False,do_rescale=True,
                do_normalize=False,patch_size=patch_size,pooling_kernel_size=pool,max_soft_tokens=70)
            processor.save_pretrained(package)
        else:write_json(package / "preprocessor_config.json", {
            "do_resize": False,
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": False,
            "do_pan_and_scan": False,
            "do_center_crop": False,
            "patch_size": patch_size,
            "pooling_kernel_size": pool,
        })

        grid = side // patch_size
        rgb = ((np.arange(side * side * 3, dtype=np.uint32) * 37 + 11 + case_index * 19) % 256).astype(np.uint8).reshape(side, side, 3)
        alternate = 255 - rgb
        patches = torch.from_numpy(patches_from_rgb(rgb, side, patch_size)).unsqueeze(0).float().requires_grad_()
        alternate_patches = torch.from_numpy(patches_from_rgb(alternate, side, patch_size)).unsqueeze(0).float()
        positions = torch.tensor([[[x, y] for y in range(grid) for x in range(grid)]], dtype=torch.long)
        if processor is not None:
            def processed(image):
                result=processor(images=[Image.fromarray(image)],return_tensors="pt")
                valid=result["image_position_ids"][0].ge(0).all(-1)
                # Fixed-resolution qualification compacts only explicit padding;
                # both engines receive the same valid patch positions.
                return result["pixel_values"][:,valid].contiguous().float(),result["image_position_ids"][:,valid].contiguous()
            patches,positions=processed(rgb)
            patches.requires_grad_()
            alternate_patches,alternate_positions=processed(alternate)
            if not torch.equal(positions,alternate_positions):raise ValueError("alternate image patch grid changed")
        if grid * grid // (pool * pool) != soft_tokens:
            raise ValueError("fixture image does not pool to the requested soft token count")
        sequences = [[1] + [63] * soft_tokens + [5 + i, 9, 2] for i in range(1)]
        ids = torch.tensor(sequences)
        mask = ids.ne(0).long()
        targets = ids.roll(-1, dims=1).clone()
        weights = mask.float()
        weights[:, -1] = 0
        weights[:, :-1] *= mask[:, 1:]
        weights[targets.eq(63)] = 0
        targets[weights.eq(0)] = -100
        fixture = {
            "batch_size": 1,
            "seq_len": ids.shape[1],
            "image_count": 1,
            "input_ids": ids.flatten().tolist(),
            "attention_mask": mask.float().flatten().tolist(),
            "targets": [x if x >= 0 else 0xFFFFFFFF for x in targets.flatten().tolist()],
            "loss_weights": weights.flatten().tolist(),
            "rgb": rgb.flatten().tolist(),
            "alternate_rgb": alternate.flatten().tolist(),
            "positions": positions.flatten().tolist(),
            "steps": 2,
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 0.001,
            "weight_decay": 0.0,
        }
        write_json(folder / "input.json", fixture)
        observed = {}

        def before_projector(module, args, kwargs):
            observed["vision_features"] = kwargs["inputs_embeds"]

        def after_projector(module, inputs, output):
            observed["projected_features"] = output
            if output.requires_grad:
                output.retain_grad()

        hooks = [
            model.model.embed_vision.register_forward_pre_hook(before_projector, with_kwargs=True),
            model.model.embed_vision.register_forward_hook(after_projector),
        ]

        def forward(pixel_values):
            logits = model(
                input_ids=ids,
                attention_mask=mask,
                pixel_values=pixel_values,
                image_position_ids=positions,
                use_cache=False,
            ).logits
            return {
                "logits": logits.detach().flatten().tolist(),
                "vision_features": observed["vision_features"].detach().flatten().tolist(),
                "projected_features": observed["projected_features"].detach().flatten().tolist(),
            }, logits

        with torch.no_grad():
            alternate_output, _ = forward(alternate_patches)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=0.001,
            weight_decay=0.0, foreach=False, fused=False,
        )
        steps = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            patches.grad = None
            output, logits = forward(patches)
            row_loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape_as(weights)
            loss = (row_loss * weights).sum() / weights.sum()
            loss.backward()
            training = {
                "text": {"step": step, "loss": loss.item()},
                "patch_gradient": patches.grad.flatten().tolist(),
                "projected_gradient": observed["projected_features"].grad.flatten().tolist(),
            }
            optimizer.step()
            steps.append({"forward": output, "training": training})
            save_file(
                {k: v.detach().cpu().contiguous().clone() for k, v in model.named_parameters()},
                str(folder / f"cpu-step-{step}.safetensors"),
            )
        with torch.no_grad():
            final_output, _ = forward(patches)
        for hook in hooks:
            hook.remove()
        vocab = config.text_config.vocab_size
        visible = mask.bool().unsqueeze(-1).expand(1, ids.shape[1], vocab).flatten().tolist()
        text_visible = (mask.bool() & ids.ne(63)).unsqueeze(-1).expand(1, ids.shape[1], vocab).flatten().tolist()
        write_json(folder / "cpu.json", {
            "pixels": patches.detach().flatten().tolist(),
            "positions": positions.flatten().tolist(),
            "alternate": alternate_output,
            "steps": steps,
            "final": final_output,
            "visible": visible,
            "text_visible": text_visible,
            "aliases": getattr(model, "all_tied_weights_keys", {}),
        })
        files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        manifest["cases"].append({"name": name, "files": files,"pixel_input_domain":pixel_domain,
            "preprocessing":"Gemma4ImageProcessorPil with explicit padding compacted" if processor is not None else "legacy raw model-value stress fixture"})
        write_json(out / "manifest.json", manifest)
        print(f"CPU {name}: raw RGB, real Gemma4 tower/embedder/text, two AdamW steps frozen", flush=True)
    if not manifest["cases"]:
        raise ValueError("no selected pipeline cases")
    source = out / "oracle-source"
    source.mkdir()
    for name in [Path(__file__).name, "qualify_multimodal_training.py"]:
        shutil.copyfile(Path(__file__).parent / name, source / name)
    upstream = (checkout / "src").resolve()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            p = Path(filename).resolve()
            if p.suffix == ".py" and p.is_relative_to(upstream):
                destination = source / p.relative_to(upstream)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, destination)
    manifest["oracle"]["source_sha256"] = {
        p.relative_to(out).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()
    }
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)


def run(args):
    root = args.fixtures.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("complete"),
        manifest.get("logit_absolute_limit"),
        manifest.get("parameter_absolute_limit"),
        manifest.get("rtol"),
        manifest.get("comparison"),
    ) != (True, LIMIT, LIMIT, 0, "strict_less_than"):
        raise ValueError("incomplete or altered pipeline acceptance policy")
    environment = oracle_env(manifest["oracle"]["cpu"])
    cases = manifest["cases"]
    names = [case["name"] for case in cases]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate frozen pipeline cases")
    if args.cases is not None:
        if not args.cases or set(args.cases) - set(names):
            raise ValueError("unknown or empty requested pipeline cases")
        cases = [case for case in cases if case["name"] in args.cases]
    for case in cases:
        if case.get("pixel_input_domain","raw_patch_values") not in {"raw_patch_values","rescaled_rgb8"}:
            raise ValueError("unsupported frozen pixel input domain")
    for name, expected in manifest["oracle"]["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"frozen CPU source changed: {name}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary_hash = digest(args.binary)
    results = []

    def persist():
        complete = len(results) == len(cases)
        write_json(out / "qualification.json", {
            "schema": 1,
            "scope": manifest["scope"],
            "absolute_limit": LIMIT,
            "rtol": 0,
            "comparison": "strict_less_than",
            "expected_cases": len(cases),
            "completed_cases": len(results),
            "complete": complete,
            "target": args.target,
            "aten_vector_width": int(environment["HIERARCHOS_ATEN_VECTOR_WIDTH"]),
            "results": results,
            "pass": complete and digest(args.binary) == binary_hash and all(r["pass"] for r in results),
            "fixtures_sha256": digest(root / "manifest.json"),
            "binary_sha256": binary_hash,
            "runner_sha256": digest(__file__),
            "pixel_input_domains":{c["name"]:c.get("pixel_input_domain","raw_patch_values") for c in cases},
        })

    persist()
    for case in cases:
        for name, expected in case["files"].items():
            if digest(root / name) != expected:
                raise ValueError(f"frozen CPU input changed: {name}")
        folder = root / case["name"]
        destination = out / case["name"]
        process = subprocess.run(
            [str(args.binary.resolve()), "--model", str(folder / "model"),
             "--fixture", str(folder / "input.json"), "--output", str(destination),
             "--pixel-domain",case.get("pixel_input_domain","raw_patch_values")],
            capture_output=True, text=True, env=environment,
        )
        (out / f"{case['name']}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            results.append({"case": case["name"], "pass": False, "exit_code": process.returncode})
            persist()
            print(f"FAIL {case['name']}: native exit {process.returncode}", flush=True)
            continue
        actual = json.loads((destination / "report.json").read_text())
        if actual.get("pixel_input_domain")!=case.get("pixel_input_domain","raw_patch_values"):
            raise ValueError("native pixel input domain differs from the frozen oracle")
        cpu = json.loads((folder / "cpu.json").read_text())
        validate_device(actual, args.target, args.device_id)
        checks = []
        diagnostics = []

        def check(label, a, b, required=True, visible=None):
            if visible is not None:
                if len(a) != len(visible) or len(b) != len(visible):
                    raise ValueError("logit shape mismatch")
                if not all(math.isfinite(v) for v in a + b):
                    raise ValueError("non-finite padded/visible logit")
                a = [v for v, keep in zip(a, visible) if keep]
                b = [v for v, keep in zip(b, visible) if keep]
            m = {"metric": label, **compare(a, b)}
            (checks if required else diagnostics).append(m)

        check("RGB patchification", actual["pixels"], cpu["pixels"])
        check("patch positions", actual["positions"], cpu["positions"])

        def forward(label, a, b):
            check(label + "/logits", a["logits"], b["logits"], visible=cpu["visible"])
            for key in ["vision_features", "projected_features"]:
                check(label + "/" + key, a[key], b[key], False)

        forward("alternate", actual["alternate"], cpu["alternate"])
        if len(actual["steps"]) != 2:
            raise ValueError("two pipeline optimizer steps required")
        for index, (a, b) in enumerate(zip(actual["steps"], cpu["steps"], strict=True), 1):
            if a["training"]["text"]["step"] != index:
                raise ValueError("wrong native optimizer step")
            forward(f"step-{index}", a["forward"], b["forward"])
            for key in ["patch_gradient", "projected_gradient"]:
                check(f"step-{index}/" + key, a["training"][key], b["training"][key], False)
            delta = abs(a["training"]["text"]["loss"] - b["training"]["text"]["loss"])
            checks.append({"metric": f"step-{index}/loss", "max_abs": delta, "pass": math.isfinite(delta) and delta < 2e-5})
            native = read_f32_tensors(destination / f"step-{index}" / "model.safetensors")
            for name, (shape, values) in read_f32_tensors(folder / f"cpu-step-{index}.safetensors").items():
                key = name if name in native else cpu["aliases"].get(name, name)
                if key not in native or native[key][0] != shape:
                    raise ValueError(f"missing/reshaped {name}")
                check(f"step-{index}/parameter/" + name, native[key][1], values)
        forward("final", actual["final"], cpu["final"])
        initial = read_f32_tensors(folder / "model" / "model.safetensors")
        trained = read_f32_tensors(destination / "step-2" / "model.safetensors")
        for label, prefix in [
            ("vision tower", "model.vision_tower."),
            ("vision embedder", "model.embed_vision."),
            ("language backbone", "model.language_model."),
        ]:
            changes = [
                abs(a - b)
                for name, (shape, values) in initial.items()
                if name.startswith(prefix)
                for a, b in zip(values, trained[name][1], strict=True)
            ]
            if not changes:
                raise ValueError(f"missing parameter group {label}")
            change = max(changes)
            checks.append({"metric": label + " parameters updated", "max_abs_change": change, "pass": change > 0.0})
        pixel_change = max(abs(value) for value in actual["steps"][0]["training"]["patch_gradient"])
        checks.append({"metric": "loss reaches image patches", "max_abs_gradient": pixel_change, "pass": pixel_change > 0.0})
        reload_values = actual["reload"]["logits"]
        final_values = actual["final"]["logits"]
        if len(reload_values) != len(final_values):
            raise ValueError("reload logit shape differs")
        reload_differences = sum(
            struct.pack("<f", a) != struct.pack("<f", b)
            for a, b in zip(reload_values, final_values)
        )
        checks.append({"metric": "native export/reload", "word_differences": reload_differences, "pass": reload_differences == 0})
        changed = max(
            abs(a - b)
            for a, b, keep in zip(
                actual["alternate"]["logits"],
                actual["steps"][0]["forward"]["logits"],
                cpu["text_visible"],
            )
            if keep
        )
        checks.append({"metric": "image affects visible text logits", "max_abs_change": changed, "pass": changed > 0.0})
        result = {
            "case": case["name"],
            "device": actual["device"],
            "pass": all(c["pass"] for c in checks),
            "checks": checks,
            "strict_intermediate_diagnostics": diagnostics,
        }
        results.append(result)
        persist()
        failures = [c for c in checks if not c["pass"]]
        print(
            f"{'PASS' if result['pass'] else 'FAIL'} {case['name']}: {len(failures)} logit/update failures; "
            f"{sum(not c['pass'] for c in diagnostics)} strict intermediate differences",
            flush=True,
        )
        for failure in failures[:8]:
            print(failure, flush=True)
    if digest(args.binary) != binary_hash:
        raise ValueError("native binary changed during qualification")
    if not all(r["pass"] for r in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p = sub.add_parser("run")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--target", choices=["gen9", "rdna3"], required=True)
    p.add_argument("--device-id", type=lambda x: int(x, 0))
    p.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()

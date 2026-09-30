"""Strict saved-module oracle on each independently base-green architecture.

Runs the common native bank test separately for interleaved A/B moment isolation.
This oracle verifies HF replacement semantics, gradients, AdamW, resume and export.
"""
from __future__ import annotations
import json
import os
import subprocess
import argparse
import time
from pathlib import Path
import torch
import verify_peft_multi_adapter as multi
import verify_peft_lora_strict as strict

ROOT = strict.ROOT
TOL = 2e-7
# Use the same pinned local oracle and dependency paths as the strict base suite.
import sys
for dependency in [ROOT / ".k3-oracle-pydeps", ROOT / ".validation-pydeps"]:
    if dependency.is_dir(): sys.path.insert(0, str(dependency))
if os.environ.get("HIERARCHOS_VALIDATION_PYDEPS"):
    sys.path.insert(0, os.environ["HIERARCHOS_VALIDATION_PYDEPS"])
sys.path.insert(0, str(strict.TRANSFORMERS_ROOT / "src"))
from safetensors.torch import load_file
from peft import PeftModel
from transformers import AutoModelForCausalLM



def main():
    strict.local_transformers_provenance()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", nargs="?", default=strict.FAMILY)
    parser.add_argument("--module", action="append", help="diagnose selected saved modules; does not replace full qualification evidence")
    args = parser.parse_args()
    report_stem = "saved-modules-selected" if args.module else "saved-modules"
    report_path = strict.FIXTURE_ROOT / (report_stem + "-report.json")
    progress_path = strict.FIXTURE_ROOT / (report_stem + "-progress.json")
    # Direct invocations need the same stale-evidence protection as the matrix
    # runner. A failed subprocess must never leave an earlier PASS collectable.
    report_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    evidence = json.loads((strict.FIXTURE_ROOT / "report.json").read_text())
    if not evidence["pass"]:
        raise RuntimeError("ordinary PEFT gate is not green; replacement qualification skipped")
    torch.set_num_threads(1)
    logits_bin = strict._native_binary()
    train_bin = strict._training_binary()
    bank_bin = ROOT / "target" / "debug" / ("peft_bank_parity.exe" if os.name == "nt" else "peft_bank_parity")
    if os.environ.get("HIERARCHOS_PEFT_USE_PREBUILT") != "1":
        multi._run(["cargo", "build", "--bin", "peft_bank_parity", "-j", "1"])
    fixture = json.loads(multi.FIXTURE_JSON.read_text())
    ids = torch.tensor(fixture["input_ids"]).reshape(fixture["batch_size"], fixture["seq_len"])
    mask = torch.tensor(fixture["attention_mask"]).reshape_as(ids)
    base = multi.load_base().eval()
    with torch.no_grad():
        hf_base = base(input_ids=ids, attention_mask=mask).logits.flatten().tolist()
    base_output = multi.FIXTURE_ROOT / "saved-modules-base-check.json"
    multi._run([str(logits_bin), "--model", str(multi.BASE_DIR), "--fixture", str(multi.FIXTURE_JSON), "--output", str(base_output)])
    base_drift = multi._max_abs(json.loads(base_output.read_text())["logits"], hf_base)
    if base_drift > TOL:
        raise RuntimeError(f"base parity not yet green: {base_drift}; PEFT skipped")
    reports = []
    modules = {
        "gpt2": ["attn.c_proj", "ln_1", "lm_head", "transformer.wte"],
        "llama": ["self_attn.o_proj", "input_layernorm", "lm_head", "model.embed_tokens"],
        "mixtral": ["self_attn.o_proj", "input_layernorm", "lm_head", "model.embed_tokens"],
        "qwen3_next": ["self_attn.o_proj", "linear_attn.out_proj", "lm_head", "model.embed_tokens"],
    }.get(strict.FAMILY)
    if modules is None:
        weights = load_file(str(strict.ADAPTER_DIR / "adapter_model.safetensors"))
        linears = sorted(key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
                         for key in weights if key.endswith(".lora_A.weight"))
        embedding = next(name for name, module in base.named_modules() if module is base.get_input_embeddings())
        head = next(name for name, module in base.named_modules() if module is base.get_output_embeddings())
        modules = [linears[0], head, embedding]
        layer_root = linears[0].split(".layers.")[0] + ".layers.0"
        norm = layer_root + (".attn_hyper_connection.hc_norm" if strict.FAMILY.startswith("qwen4_exp") else ".input_layernorm")
        if norm in dict(base.named_modules()):
            modules.insert(1, norm)
    if args.module:
        unknown = set(args.module) - set(modules)
        if unknown:
            parser.error(f"unknown saved modules: {sorted(unknown)}; available: {modules}")
        modules = [module for module in modules if module in args.module]
    # Keep transient native outputs deliberately short on Windows.  The
    # architecture fixture root plus a full modules_to_save name can otherwise
    # push checkpoint/bank paths far enough that ordinary file creation fails
    # with ERROR_ACCESS_DENIED even though the model and adapter are valid.
    # Canonical qualification reports remain in FIXTURE_ROOT; this directory is
    # only scratch state for the current process/run.
    run_work_root = ROOT / ".peft-validation-work" / strict.FAMILY / f"{os.getpid()}-{time.time_ns()}"
    run_work_root.mkdir(parents=True, exist_ok=True)
    for module_index, saved in enumerate(modules):
        multi.SAVED = saved
        multi.MULTI_ROOT = run_work_root / f"m{module_index}"
        multi.MULTI_ROOT.mkdir(parents=True, exist_ok=True)
        a, b = multi.MULTI_ROOT / "adapter-a", multi.MULTI_ROOT / "adapter-b"
        multi._write_adapter(a, 101)
        multi._write_adapter(b, 202)
        reference = PeftModel.from_pretrained(multi.load_base(), a, adapter_name="adapter_a").eval()
        reference.load_adapter(b, adapter_name="adapter_b")
        report = {"module": saved, "base_max_abs": base_drift}
        for name in ["adapter_a", "adapter_b"]:
            reference.set_adapter(name)
            with torch.no_grad():
                expected = reference(input_ids=ids, attention_mask=mask).logits.flatten().tolist()
            native = multi._native_logits(logits_bin, a, b, name)
            report[name + "_max_abs"] = multi._max_abs(native["logits"], expected)
        native = multi._native_logits(logits_bin, a, b, "adapter_a", disabled=True)
        report["disabled_base_max_abs"] = multi._max_abs(native["logits"], hf_base)
        # Independently train both named payloads against the HF oracle.
        for label, adapter in [("a", a), ("b", b)]:
            model = PeftModel.from_pretrained(multi.load_base(), adapter, is_trainable=True).train()
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([p for _, p in params], lr=0.001, eps=0.001, weight_decay=0.0)
            gradients = {}
            for step in range(2):
                optimizer.zero_grad()
                model(input_ids=ids, attention_mask=mask, labels=ids).loss.backward()
                if step == 0:
                    for name, param in params:
                        assert param.grad is not None, name
                        key = strict._saved_lora_key(name).replace(".modules_to_save.default.", ".")
                        gradients[key] = param.grad.detach().clone()
                optimizer.step()
            hf_dir = multi.MULTI_ROOT / (label + "-hf-trained")
            model.save_pretrained(hf_dir)
            outputs = {}
            checkpoint = multi.MULTI_ROOT / (label + "-checkpoint")
            # Repeated runs preserve old evidence by using a new checkpoint directory.
            checkpoint = checkpoint.with_name(checkpoint.name + "-" + str(time.time_ns()))
            for steps in [1, 2]:
                out = multi.MULTI_ROOT / f"{label}-native-{steps}"
                args = [str(train_bin), "--model", str(multi.BASE_DIR), "--fixture", str(multi.FIXTURE_ROOT / "training.json"), "--output", str(out), "--lora-adapter", str(adapter), "--steps", str(steps)]
                if steps == 1:
                    args += ["--gradient-output", str(multi.MULTI_ROOT / (label + "-gradients.json")), "--training-checkpoint", str(checkpoint), "--base-output", str(multi.MULTI_ROOT / (label + "-base-after"))]
                multi._run(args)
                outputs[steps] = load_file(str(out / "adapter_model.safetensors"))
            resumed = multi.MULTI_ROOT / (label + "-resumed")
            multi._run([str(train_bin), "--model", str(multi.BASE_DIR), "--fixture", str(multi.FIXTURE_ROOT / "training.json"), "--output", str(resumed), "--resume-lora", str(checkpoint), "--steps", "1"])
            hf_weights = load_file(str(hf_dir / "adapter_model.safetensors"))
            initial = load_file(str(adapter / "adapter_model.safetensors"))
            report[label + "_trainable_keys"] = set(initial) == set(gradients)
            native_grad = {key: torch.tensor(value).reshape_as(gradients[key]) for key, value in json.loads((multi.MULTI_ROOT / (label + "-gradients.json")).read_text()).items()}
            gradient_max_abs, gradient_worst_tensor = strict._tensor_map_max_abs(native_grad, gradients)
            report[label + "_gradient_max_abs"] = gradient_max_abs
            report[label + "_gradient_worst_tensor"] = gradient_worst_tensor
            adamw_max_abs, adamw_worst_tensor = strict._tensor_map_max_abs(outputs[2], hf_weights)
            report[label + "_adamw_max_abs"] = adamw_max_abs
            report[label + "_adamw_worst_tensor"] = adamw_worst_tensor
            report[label + "_resume_max_abs"] = strict._tensor_map_max_abs(outputs[2], load_file(str(resumed / "adapter_model.safetensors")))[0]
            report[label + "_frozen_base_max_abs"] = strict._tensor_map_max_abs(load_file(str(multi.BASE_DIR / "model.safetensors")), load_file(str(multi.MULTI_ROOT / (label + "-base-after") / "model.safetensors")))[0]
            # Native-created adapter files must load into HF without layout fixes.
            native_model = PeftModel.from_pretrained(multi.load_base(), multi.MULTI_ROOT / f"{label}-native-2").eval()
            with torch.no_grad():
                hf_native = native_model(input_ids=ids, attention_mask=mask).logits.flatten().tolist()
            native_report = json.loads((multi.MULTI_ROOT / f"{label}-native-2" / "training_report.json").read_text())
            report[label + "_hf_reload_max_abs"] = multi._max_abs(native_report["logits_after"], hf_native)
        # Exercise interleaved A/B training, gradients and moments on THIS graph,
        # then reload both adapters together and compare with independent runs.
        # Keep the native bank probe substantially shorter than MULTI_ROOT on
        # Windows.  The bank binary creates several nested checkpoint staging
        # directories beneath its output root; placing those under the already
        # nested per-module scratch directory can push later atomic checkpoint
        # publishes into Windows path/AV edge cases (observed as os error 5).
        bank_root = ROOT / ".peft-bank-work"
        bank_root.mkdir(parents=True, exist_ok=True)
        bank_output = bank_root / f"{strict.FAMILY}-{os.getpid()}-{module_index}-{time.time_ns()}"
        multi._run([str(bank_bin), str(multi.BASE_DIR), str(multi.FIXTURE_ROOT / "training.json"), str(a), str(b), str(bank_output)])
        report["bank_isolation"] = json.loads((bank_output / "report.json").read_text())["pass"]
        report["pass"] = all(
            v if isinstance(v, bool) else v <= TOL
            for k, v in report.items()
            if k != "module" and not k.endswith("_worst_tensor")
        ) and report["a_frozen_base_max_abs"] == 0.0 and report["b_frozen_base_max_abs"] == 0.0
        reports.append(report)
        progress_path.write_text(json.dumps({"complete": len(reports) == len(modules), "expected_modules": modules, "results": reports}, indent=2))
        print(json.dumps(report, indent=2), flush=True)

    report_path.write_text(json.dumps(reports, indent=2))
    if not all(report["pass"] for report in reports):
        raise RuntimeError(f"strict modules_to_save parity failed; see {report_path}")

if __name__ == "__main__":
    main()

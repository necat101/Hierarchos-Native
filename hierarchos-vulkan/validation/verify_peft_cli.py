"""Native CLI training, percentage budget, HF config and automatic merge smoke.

Uses the already-qualified deterministic GPT-2 fixture; never downloads a model.
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
import sys
import struct
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "hierarchos-vulkan/.peft-oracle-fixtures/gpt2-strict/base"


def main():
    cli = Path(sys.argv[1]).resolve()
    out = Path(tempfile.mkdtemp(prefix="peft-cli-", dir=ROOT / "hierarchos-vulkan/.peft-oracle-fixtures"))
    base = out / "base"
    shutil.copytree(BASE, base)
    assets = ROOT / "hierarchos-native-cli/tests/fixtures"
    shutil.copyfile(assets / "peft_tokenizer.json", base / "tokenizer.json")
    shutil.copyfile(assets / "peft_generation_config.json", base / "generation_config.json")
    data = out / "train.jsonl"
    data.write_text(json.dumps({"input_ids": [1, 7, 3, 11, 2]}) + "\n")

    def run(destination, *args):
        command = [str(cli), "transformer-finetune", "--model-path", str(base),
                   "--train", str(data), "--out-dir", str(out / destination),
                   "--seq-len", "4", "--epochs", "1", "--lr", "0.001", *args]
        result = subprocess.run(command, text=True, capture_output=True)
        (out / (destination + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stderr

    log = run("budget", "--peft", "--peft-parameter-percent", "10", "--lora-merge")
    assert "achieved_percent=" in log, log
    budget_config = json.loads((out / "budget/adapter_config.json").read_text())
    assert budget_config["r"] >= 1
    def tensor_count(path):
        with path.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        import math
        return sum(math.prod(tensor["shape"]) for name, tensor in header.items() if name != "__metadata__")
    base_count = tensor_count(base / "model.safetensors")
    adapter_count = tensor_count(out / "budget/adapter_model.safetensors")
    assert adapter_count <= base_count * 0.10
    assert adapter_count / budget_config["r"] * (budget_config["r"] + 1) > base_count * 0.10
    for file in ["budget/adapter_model.safetensors", "budget/training-state/hierarchos_adapter_optimizer.safetensors",
                 "budget/merged-model/model.safetensors", "budget/merged-model/config.json"]:
        assert (out / file).is_file(), file

    config = out / "config.json"
    config.write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2,
        "lora_alpha": 4, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": True,
        "target_modules": ["c_attn"], "layers_to_transform": [0], "layers_pattern": "h",
        "rank_pattern": {}, "alpha_pattern": {}}))
    run("configured", "--peft-config", str(config))
    saved = json.loads((out / "configured/adapter_config.json").read_text())
    assert saved["r"] == 2 and saved["layers_to_transform"] == [0]
    run("direct", "--peft", "--lora-r", "2", "--lora-alpha", "4",
        "--lora-target-modules", "c_attn,c_proj", "--lora-exclude-modules", "mlp.c_proj",
        "--lora-layers-to-transform", "0", "--lora-layers-pattern", "h",
        "--lora-rank-pattern", '{"attn.c_attn": 3}', "--lora-alpha-pattern", '{"attn.c_attn": 6}')
    direct = json.loads((out / "direct/adapter_config.json").read_text())
    assert direct["rank_pattern"] == {"attn.c_attn": 3}
    assert direct["exclude_modules"] == ["mlp.c_proj"]
    run("resumed", "--peft-resume", str(out / "configured/training-state"), "--lora-model-merge")
    run("uninterrupted", "--peft-config", str(config), "--epochs", "2")
    def tensor_payload(path):
        with path.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
            data = stream.read()
        return {name: (entry["dtype"], entry["shape"], data[slice(*entry["data_offsets"])])
                for name, entry in header.items() if name != "__metadata__"}
    for name in ["adapter_model.safetensors", "training-state/hierarchos_adapter_optimizer.safetensors"]:
        assert tensor_payload(out / "resumed" / name) == tensor_payload(out / "uninterrupted" / name), name
    run("full", "--full-finetune")
    assert (out / "full/model.safetensors").is_file()

    # Compare auto-merged logits to the saved unmerged native adapter.
    binary = ROOT / "hierarchos-vulkan/target/debug/hierarchos-vulkan-transformer-logits.exe"
    fixture = BASE.parent / "fixture.json"
    outputs = []
    for name, model, extra in [
        ("unmerged", base, ["--lora-adapter", str(out / "budget")]),
        ("merged", out / "budget/merged-model", []),
    ]:
        path = out / (name + ".json")
        subprocess.run([str(binary), "--model", str(model), "--fixture", str(fixture),
                        "--output", str(path), *extra], check=True, capture_output=True)
        outputs.append(json.loads(path.read_text())["logits"])
    drift = max(abs(a-b) for a, b in zip(*outputs))
    assert drift <= 2e-7, drift
    generations = []
    for model, extra in [(base, ["--peft-adapter", str(out / "budget")]),
                         (out / "budget/merged-model", [])]:
        result = subprocess.run([str(cli), "transformer-generate", "--model-path", str(model),
                                 "--prompt", "hello", "--max-new-tokens", "1", "--no-do-sample",
                                 *extra], text=True, capture_output=True, check=True)
        generations.append(result.stdout)
    assert generations[0] == generations[1], generations
    # Production CLI enforcement, not just validator executables that bypass
    # registry gates: train and load promoted norm/embedding replacement banks.
    saved_smokes = []
    for family in ["gemma3", "gemma4"]:
        gemma_base = out / (family + "-base")
        shutil.copytree(BASE.parents[1] / (family + "-strict/base"), gemma_base)
        shutil.copyfile(assets / "peft_tokenizer.json", gemma_base / "tokenizer.json")
        shutil.copyfile(assets / "peft_generation_config.json", gemma_base / "generation_config.json")
        destination = out / (family + "-saved")
        result = subprocess.run([str(cli), "transformer-finetune", "--model-path", str(gemma_base),
            "--train", str(data), "--out-dir", str(destination), "--seq-len", "4", "--epochs", "1",
            "--lr", "0.001", "--peft", "--lora-rank", "2", "--lora-alpha", "4",
            "--lora-target-modules", "q_proj,v_proj",
            "--lora-modules-to-save", "model.layers.0.input_layernorm",
            "--lora-modules-to-save", "model.embed_tokens"], text=True, capture_output=True)
        (out / (family + "-saved.log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        assert (destination / "adapter_model.safetensors").is_file()
        assert (destination / "training-state/hierarchos_adapter_optimizer.safetensors").is_file()
        result = subprocess.run([str(cli), "transformer-generate", "--model-path", str(gemma_base),
            "--peft-adapter", str(destination), "--prompt", "hello", "--max-new-tokens", "1",
            "--no-do-sample"], text=True, capture_output=True)
        (out / (family + "-saved-generation.log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        saved_smokes.append(family)
    report = {"pass": True, "auto_merge_max_abs": drift,
              "resume_weights_and_moments_exact": True, "model_merge_alias": True,
              "merged_generation_matches_adapter": True, "saved_module_cli_smokes": saved_smokes,
              "output": str(out)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

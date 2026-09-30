"""Exercise the user CLI and GUI command builder with a tiny Falcon checkpoint."""
import argparse
import json
import os
import subprocess

import torch
from verify_falcon_h1 import ATOL, ROOT, REPO, compare, roundtrip, transformers

from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = ROOT / ".falcon-h1-fixtures"
    source = root / "gated_after"
    cli = REPO / "hierarchos-native-cli/target/debug/hierarchos-native-cli.exe"
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(41)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(source / "tokenizer.json"))
    ids = [1, 5, 9, 3, 7, 2, 6, 8]
    dataset = root / "frontend-data.jsonl"
    dataset.write_text(json.dumps({"input_ids": ids}) + "\n")
    report = {"passed": False}
    (root / "frontend-report.json").write_text(json.dumps(report, indent=2))
    model = transformers.FalconH1ForCausalLM.from_pretrained(source, attn_implementation="eager").train()
    current = source
    for mode, name, extra in [("transformer-train", "cli-trained", []),
                              ("transformer-finetune", "cli-finetuned", ["--full-finetune"])]:
        destination = root / name
        command = [str(cli), mode, "--model-path", str(current), "--train", str(dataset),
                   "--out-dir", str(destination), "--epochs", "2", "--batch-size", "1",
                   "--seq-len", "7", "--lr", "0.0002", *extra]
        subprocess.run(command, check=True, cwd=REPO)
        # A new CLI process deliberately starts a new optimizer trajectory.
        # Match the CLI's existing optimizer groups. Transformer outer norms
        # and linear biases have decay=false; Falcon's packed Mamba parameters
        # (including its gated norm/conv bias) have decay=true.
        no_decay = []
        decay = []
        for key, parameter in model.named_parameters():
            outer_norm = key.endswith(("input_layernorm.weight", "pre_ff_layernorm.weight", "final_layernorm.weight"))
            linear_bias = key.endswith(".bias") and not key.endswith("mamba.conv1d.bias")
            (no_decay if outer_norm or linear_bias else decay).append(parameter)
        optimizer = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], lr=0.0002, eps=1e-8,
                                      betas=(0.9, 0.999), weight_decay=0.01, foreach=False)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.tensor([ids[:-1]]), use_cache=False).logits
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor(ids[1:])).backward()
            optimizer.step()
        saved = load_file(destination / "model.safetensors")
        worst = (0.0, "")
        for key, parameter in model.named_parameters():
            delta = compare(saved[key], parameter.detach(), f"{name}/{key}")
            worst = max(worst, (delta, key))
        report[name] = {"parameter_max_abs": worst[0], "worst_parameter": worst[1],
                        "roundtrip": roundtrip(destination, destination)}
        subprocess.run([str(cli), "infer", "--model-path", str(destination), "--prompt", "t1 t5 t9",
                        "--max-new-tokens", "5", "--no-do-sample", "--use-cache"], check=True, cwd=REPO)
        current = destination
    if args.gui:
        subprocess.run(["cargo", "test", "--manifest-path", str(REPO / "hierarchos-gui/Cargo.toml"),
                        "-j", "1", "falcon_h1_native_training_finetuning_and_generation", "--", "--ignored", "--nocapture"],
                       check=True, cwd=REPO,
                       env={**os.environ, "FALCON_H1_SMOKE_ROOT": str(root), "HIERARCHOS_NATIVE_CLI": str(cli)})
        for gui_name, cli_name in [("gui-trained", "cli-trained"), ("gui-finetuned", "cli-finetuned")]:
            gui_weights = load_file(root / gui_name / "model.safetensors")
            cli_weights = load_file(root / cli_name / "model.safetensors")
            assert gui_weights.keys() == cli_weights.keys()
            assert all(torch.equal(value, cli_weights[key]) for key, value in gui_weights.items())
            report[gui_name] = {"cli_parameter_max_abs": 0.0, "roundtrip": roundtrip(root / gui_name, root / gui_name)}
    report["atol"] = ATOL
    report["passed"] = True
    (root / "frontend-report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

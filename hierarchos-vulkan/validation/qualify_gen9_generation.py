"""Freeze CPU generation from qualified packages, then test both storage modes.

Cached decode and full-prefix execution each use their own independent CPU
expectations. CPU preparation never executes a native probe or changes weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from qualify_gen9_architectures import LIMIT, compare_logits, digest, validate_residency, write_json

DEFAULT_PROFILES = ["gpt2", "llama", "mistral", "mixtral", "qwen2", "qwen3_next",
    "gpt_oss", "smollm3", "qwen2_5_gqa", "qwen2_5_sliding_tied",
    "qwen3_5_full", "qwen3_5_linear", "qwen3_5_mixed", "qwen3_5_moe",
    "qwen4_exp_linear", "qwen4_exp_qsa", "qwen4_exp_ple", "qwen4_exp_mixed_ple"]


def verify_package(root, case):
    for name, expected in case["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"frozen CPU package changed: {name}")


def prepare(args):
    import torch
    import verify_hf_logits as reference
    import transformers

    torch.set_num_threads(1)
    root = args.fixtures.resolve()
    source = json.loads((root / "manifest.json").read_text())
    cases = {case["family"]: case for case in source["cases"] if case["case"] == "unmasked"}
    requested = args.families or DEFAULT_PROFILES
    if len(requested) != len(set(requested)) or set(requested) - cases.keys():
        raise ValueError("duplicate or missing generation profiles")
    if args.steps < 2:
        raise ValueError("generation qualification needs at least two steps")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "absolute_limit": LIMIT, "rtol": 0,
        "comparison": "strict_less_than", "steps": args.steps,
        "source_manifest_sha256": digest(root / "manifest.json"),
        "fixture_root": str(root), "oracle": source["oracle"],
        "generation_generator_sha256": digest(__file__),
        "requested_profiles": requested, "complete": False, "cases": []}
    for family in requested:
        case = cases[family]
        verify_package(root, case)
        package = root / case["package"]
        config = json.loads((package / "config.json").read_text())
        model_class = getattr(transformers, config["architectures"][0])
        model = model_class.from_pretrained(package, local_files_only=True,
                                            attn_implementation="eager").eval()
        prompt = torch.tensor([[1, 5, 9, 3, 7]], dtype=torch.long)
        folder = out / family
        folder.mkdir()
        write_json(folder / "input.json", {"input_ids": prompt.flatten().tolist(),
                                          "max_new_tokens": args.steps})
        expected = {}
        for cached in [False, True]:
            if family.startswith("qwen3_5") or family.startswith("qwen4_exp"):
                if family.startswith("qwen3_5"):
                    from verify_qwen35_generation import hf_generation
                else:
                    from verify_qwen4_exp_generation import hf_generation
                sequence, logits = hf_generation(model, prompt, cached, args.steps)
            else:
                with torch.no_grad():
                    generated = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                        max_new_tokens=args.steps, do_sample=False, use_cache=cached,
                        eos_token_id=None, pad_token_id=0, output_logits=True,
                        return_dict_in_generate=True)
                sequence = generated.sequences.flatten().tolist()
                logits = generated.logits
            if len(logits) != args.steps:
                raise ValueError(f"{family}: incomplete CPU generation")
            bits = [step.detach().float().contiguous().view(torch.uint32).flatten().tolist() for step in logits]
            expected[str(cached).lower()] = {"sequence": sequence, "bits": bits}
            for step in bits:
                compare_logits(step, step, [True] * len(step))
        write_json(folder / "cpu.json", expected)
        manifest["cases"].append({"family": family, "package": case["package"],
            "input": f"{family}/input.json", "cpu": f"{family}/cpu.json",
            "files": {p.relative_to(out).as_posix(): digest(p) for p in sorted(folder.iterdir())}})
        write_json(out / "manifest.json", manifest)
        print(f"CPU generation {family}: {args.steps} steps, cached and full-prefix frozen", flush=True)
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)


def run(args):
    root = args.fixtures.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete") or not manifest["cases"]:
        raise ValueError("CPU generation preparation is incomplete")
    if (manifest["absolute_limit"], manifest["rtol"], manifest["comparison"]) != (LIMIT, 0, "strict_less_than"):
        raise ValueError("generation acceptance policy differs")
    packages = args.packages.resolve() if args.packages else Path(manifest["fixture_root"])
    if digest(packages / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("generation source manifest changed")
    package_cases = {case["family"]: case for case in json.loads((packages / "manifest.json").read_text())["cases"]
                     if case["case"] == "unmasked"}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    result = {"schema": 1, "target": args.target, "pass": False, "absolute_limit": LIMIT, "rtol": 0,
        "comparison": "strict_less_than", "binary": str(binary),
        "binary_sha256": digest(binary), "manifest_sha256": digest(root / "manifest.json"), "records": []}
    for case in manifest["cases"]:
        family = case["family"]
        verify_package(packages, package_cases[family])
        for name, expected in case["files"].items():
            if digest(root / name) != expected:
                raise ValueError(f"frozen CPU generation changed: {name}")
        expected = json.loads((root / case["cpu"]).read_text())
        for cached in [False, True]:
            mode_words = {}
            for mode in ["resident", "streaming"]:
                label = f"{family}-{'cache' if cached else 'full'}-{mode}"
                report_path = out / (label + ".json")
                command = [str(binary), "--model", str(packages / case["package"]),
                    "--fixture", str(root / case["input"]), "--output", str(report_path),
                    "--use-cache", str(cached).lower()]
                if mode == "streaming":
                    command.append("--stream-parameters")
                record = {"family": family, "use_cache": cached, "mode": mode,
                          "command": command, "pass": False}
                started = time.monotonic()
                try:
                    completed = subprocess.run(command, capture_output=True, text=True,
                        timeout=args.timeout, env=dict(os.environ))
                    (out / (label + ".log")).write_text(completed.stdout + completed.stderr)
                    record["exit"] = completed.returncode
                    if completed.returncode:
                        raise RuntimeError(f"native generation exited {completed.returncode}")
                    report = json.loads(report_path.read_text())
                    validate_residency(report, mode, args.target, args.expected_device_id)
                    reference = expected[str(cached).lower()]
                    if report["sequence"] != reference["sequence"]:
                        raise ValueError("generated token sequence differs from CPU")
                    actual = report["logits_bits"]
                    if len(actual) != len(reference["bits"]):
                        raise ValueError("incomplete native logit steps")
                    steps = [compare_logits(a, b, [True] * len(b))
                        for a, b in zip(actual, reference["bits"], strict=True)]
                    record.update({"steps": steps, "max_abs": max(step["max_abs"] for step in steps),
                        "pass": all(step["pass"] for step in steps), "sequence": report["sequence"],
                        "streamed_linears": report["streamed_linears"]})
                    record.update({key: report[key] for key in ["device", "vendor_id", "device_id"]})
                    mode_words[mode] = actual
                    if mode == "streaming":
                        differences = sum(a != b for a, b in zip(
                            [word for step in actual for word in step],
                            [word for step in mode_words["resident"] for word in step], strict=True))
                        record["resident_word_differences"] = differences
                        record["pass"] = record["pass"] and differences == 0
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                    record["pass"] = False
                    record["error"] = str(error)
                record["elapsed_seconds"] = time.monotonic() - started
                result["records"].append(record)
                print(f"{'PASS' if record['pass'] else 'FAIL'} {label}: {record.get('max_abs', record.get('error'))}", flush=True)
                write_json(out / "qualification.json", result)
    if digest(binary) != result["binary_sha256"]:
        raise ValueError("generation probe changed during qualification")
    result["pass"] = bool(result["records"]) and all(record["pass"] for record in result["records"])
    write_json(out / "qualification.json", result)
    return 0 if result["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+")
    p.add_argument("--steps", type=int, default=4)
    p = commands.add_parser("run")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--packages", type=Path, help="relocated frozen source packages (manifest must still match)")
    p.add_argument("--target", choices=["gen9", "rdna3"], default="gen9")
    p.add_argument("--expected-device-id", type=lambda value: int(value, 0))
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

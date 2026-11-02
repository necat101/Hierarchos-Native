"""Freeze independent CPU logits, then qualify resident and streamed Gen9 graphs.

Preparation runs no native executable. Both storage modes subsequently consume
the same hash-bound packages, inputs and CPU words. The architecture registry
is reported separately from measured numerical coverage.
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
import time

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2e-7


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def words(values):
    return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


def values(bits):
    if any(type(value) is not int or not 0 <= value <= 0xffffffff for value in bits):
        raise ValueError("invalid FP32 word")
    return [struct.unpack("<f", struct.pack("<I", word))[0] for word in bits]


def compare_logits(actual_bits, expected_bits, visible):
    if not actual_bits or len(actual_bits) != len(expected_bits) or len(visible) != len(expected_bits):
        raise ValueError("empty or inconsistent logit coverage")
    actual, expected = values(actual_bits), values(expected_bits)
    if not all(math.isfinite(value) for value in actual + expected):
        raise ValueError("non-finite logit")
    indices = [index for index, valid in enumerate(visible) if valid]
    if not indices:
        raise ValueError("no observable query rows")
    deltas = [(abs(actual[i] - expected[i]), i) for i in indices]
    worst, index = max(deltas)
    return {"values": len(indices), "excluded_values": len(visible) - len(indices),
            "max_abs": worst, "worst_index": index,
            "word_differences": sum(actual_bits[i] != expected_bits[i] for i in indices),
            "failing_values": sum(delta >= LIMIT for delta, _ in deltas),
            "pass": worst < LIMIT}


def validate_residency(report, mode, target="gen9", expected_device_id=None):
    if target == "gen9":
        if "HD Graphics 520" not in report.get("device", ""):
            raise ValueError(f"unexpected Gen9 target: {report.get('device')}")
        if "vendor_id" in report and (report["vendor_id"], report.get("device_id")) != (0x8086, 0x1916):
            raise ValueError("physical GPU PCI identity differs from the Gen9 target")
    elif target == "rdna3":
        if report.get("vendor_id") != 0x1002 or type(report.get("device_id")) is not int:
            raise ValueError("RDNA3 handoff requires a physical AMD adapter and its PCI identity")
    else:
        raise ValueError(f"unknown qualification target: {target}")
    if expected_device_id is not None and report.get("device_id") != expected_device_id:
        raise ValueError("physical GPU PCI device differs from the requested adapter")
    count = report.get("streamed_linears")
    if type(count) is not int:
        raise ValueError("missing streamed-linear coverage")
    if mode == "resident":
        if count != 0 or report.get("source_first_parameters") or report.get("loading_linear_package_io"):
            raise ValueError("resident request executed package-backed graph linears")
    elif mode == "streaming":
        if count <= 0:
            raise ValueError("streaming request did not execute any streamed linears")
    else:
        raise ValueError(f"unknown residency: {mode}")


def prepare(args):
    import torch
    import verify_hf_logits as reference
    from generate_supported_architectures import _ordered_registry
    from qualify_parameter_streaming import FAMILIES as rdna_profiles

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Retain the validation adapters as well as the upstream revision. Kimi's
    # independent CPU primitive adapters and remote model graph are both inputs.
    source_files = {}
    source_dir = out / "_oracle"
    source_dir.mkdir()
    for name in [Path(__file__).name, "verify_hf_logits.py", "verify_falcon_h1.py",
                 "kimi_k3_reference.py", "generate_supported_architectures.py", "qualify_parameter_streaming.py"]:
        destination = source_dir / name
        shutil.copyfile(Path(__file__).parent / name, destination)
        source_files[destination.relative_to(out).as_posix()] = digest(destination)
    from kimi_k3_reference import find_kimi_k3_remote_source
    remote = find_kimi_k3_remote_source()
    if remote:
        for source in sorted(remote.rglob("*.py")):
            destination = source_dir / "kimi_remote" / source.relative_to(remote)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            source_files[destination.relative_to(out).as_posix()] = digest(destination)
    torch.manual_seed(0x48494552)
    torch.set_num_threads(1)
    models = reference.tiny_models() + reference.production_width_models()
    # The existing strict PEFT matrix has three Falcon-H1 branch controls.
    from verify_falcon_h1 import tiny
    models.extend((name, tiny(norm=norm, before=before), False) for name, norm, before in [
        ("falcon_h1_gated_after", True, False),
        ("falcon_h1_gated_before", True, True),
        ("falcon_h1_silu", False, False),
    ])
    names = {name for name, _, _ in models}
    if "qwen3_next" not in names:
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        config = Qwen3NextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=16, attention_dropout=0.0, use_cache=False,
            tie_word_embeddings=True, attn_implementation="eager", head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=2, linear_conv_kernel_dim=3, num_experts=2,
            num_experts_per_tok=2, moe_intermediate_size=16,
            shared_expert_intermediate_size=16, mlp_only_layers=[0, 1],
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0})
        models.append(("qwen3_next", Qwen3NextForCausalLM(config), False))
    names = {name for name, _, _ in models}
    if args.families:
        unknown = set(args.families) - names
        if unknown:
            raise ValueError(f"missing CPU reference profiles: {sorted(unknown)}")
        models = [entry for entry in models if entry[0] in args.families]
    canonical, aliases = _ordered_registry((ROOT / "hierarchos-vulkan/src/transformer.rs").read_text())
    source_root = reference.TRANSFORMERS_SOURCE.parent
    oracle = {"transformers_root": str(source_root),
              "transformers_revision": subprocess.check_output(
                  ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
              "transformers_status": subprocess.check_output(
                  ["git", "-C", str(source_root), "status", "--porcelain"], text=True).splitlines(),
              "torch": torch.__version__, "torch_revision": torch.version.git_version,
              "cpu": torch.backends.cpu.get_cpu_capability(), "threads": torch.get_num_threads(),
              "reference_generator_sha256": digest(reference.__file__),
              "qualification_generator_sha256": digest(__file__),
              "source_files_sha256": source_files}
    manifest = {"schema": 1, "absolute_limit": LIMIT, "comparison": "strict_less_than", "rtol": 0,
                "oracle": oracle, "registry": {"canonical": canonical, "aliases": aliases},
                "rdna_verified_profiles": rdna_profiles,
                "missing_rdna_references": sorted(set(rdna_profiles) - names), "cases": []}
    for family, model, token_types in models:
        model.eval()
        for padded in ([False] if family in reference.KIMI_K3_ORACLE_FAMILIES else [False, True]):
            case = "padded" if padded else "unmasked"
            folder = out / family / case
            package = folder / "model"
            folder.mkdir(parents=True)
            reference.save_reference_checkpoint(family, model, package)
            ids = torch.tensor([[1, 5, 9, 3, 7, 2]], dtype=torch.long)
            mask = torch.ones_like(ids)
            if family in reference.WIDE_LOGIT_FAMILIES and not padded:
                ids = torch.cat([ids, ids.flip(1)])
                mask = torch.ones_like(ids)
            if padded:
                mask[0, -2:] = 0
                ids = torch.cat([ids, ids.flip(1)])
                mask = torch.cat([mask, torch.tensor([[0, 1, 1, 1, 1, 1]])])
            fixture = {"batch_size": ids.shape[0], "seq_len": ids.shape[1],
                       "input_ids": ids.flatten().tolist(),
                       "attention_mask": mask.float().flatten().tolist()}
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if token_types:
                types = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand_as(ids)
                kwargs["token_type_ids"] = types
                fixture["token_type_ids"] = types.flatten().tolist()
            encoder_decoder = model.config.is_encoder_decoder
            if encoder_decoder:
                encoder_ids = torch.tensor([[4, 8, 12, 6, 2]])
                encoder_mask = torch.tensor([[1, 1, 1, 0, 0] if padded else [1] * 5])
                if padded:
                    encoder_ids = torch.cat([encoder_ids, encoder_ids.flip(1)])
                    encoder_mask = torch.cat([encoder_mask, torch.tensor([[0, 1, 1, 1, 1]])])
                fixture["encoder_input_ids"] = encoder_ids.flatten().tolist()
                fixture["encoder_attention_mask"] = encoder_mask.float().flatten().tolist()
                kwargs = {"input_ids": encoder_ids, "attention_mask": encoder_mask,
                          "decoder_input_ids": ids, "decoder_attention_mask": mask}
            with torch.no_grad():
                logits = model(**kwargs).logits.detach().float().cpu()
            vocab = reference.model_vocab_size(model)
            expected = {"bits": logits.contiguous().view(torch.uint32).flatten().tolist(),
                        "visible": mask.bool().reshape(-1, 1).expand(-1, vocab).flatten().tolist(),
                        "vocab_size": vocab}
            compare_logits(expected["bits"], expected["bits"], expected["visible"])
            write_json(folder / "input.json", fixture)
            write_json(folder / "cpu.json", expected)
            files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}
            manifest["cases"].append({"family": family, "case": case,
                "model_type": model.config.model_type, "encoder_decoder": encoder_decoder,
                "package": package.relative_to(out).as_posix(),
                "input": (folder / "input.json").relative_to(out).as_posix(),
                "cpu": (folder / "cpu.json").relative_to(out).as_posix(), "files": files})
            write_json(out / "manifest.json", manifest)
            print(f"CPU {family}/{case}: {len(expected['bits'])} words frozen", flush=True)
    print(f"Prepared {len(manifest['cases'])} independent CPU cases", flush=True)


def run(args):
    # Compare storage modes in a stable order even if the caller reverses flags.
    args.modes = [mode for mode in ("resident", "streaming") if mode in args.modes]
    if not args.modes:
        raise ValueError("empty residency coverage")
    root = args.fixtures.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("absolute_limit"), manifest.get("rtol"), manifest.get("comparison")) != (LIMIT, 0, "strict_less_than"):
        raise ValueError("fixture acceptance policy differs")
    for name, expected_hash in manifest["oracle"].get("source_files_sha256", {}).items():
        if digest(root / name) != expected_hash:
            raise ValueError(f"frozen CPU source changed: {name}")
    cases = manifest["cases"]
    if args.families:
        missing = set(args.families) - {case["family"] for case in cases}
        if missing:
            raise ValueError(f"unknown profiles: {sorted(missing)}")
        cases = [case for case in cases if case["family"] in args.families]
    if not cases:
        raise ValueError("empty qualification matrix")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    target = getattr(args, "target", "gen9")
    result = {"schema": 1, "target": target, "pass": False, "absolute_limit": LIMIT, "rtol": 0,
              "comparison": "strict_less_than", "binary": str(binary),
              "binary_sha256": digest(binary), "manifest_sha256": digest(root / "manifest.json"),
              "fixture_root": str(root), "modes": args.modes, "records": []}
    for case in cases:
        for name, expected_hash in case["files"].items():
            if digest(root / name) != expected_hash:
                raise ValueError(f"frozen CPU input changed: {name}")
        expected = json.loads((root / case["cpu"]).read_text())
        mode_words = {}
        for mode in args.modes:
            label = f"{case['family']}-{case['case']}-{mode}"
            report_path = out / (label + ".json")
            command = [str(binary), "--model", str(root / case["package"]),
                       "--fixture", str(root / case["input"]), "--output", str(report_path)]
            if mode == "streaming":
                command.append("--stream-parameters")
            started = time.monotonic()
            record = {"family": case["family"], "case": case["case"], "mode": mode,
                      "command": command, "pass": False}
            try:
                environment = dict(os.environ, HIERARCHOS_VULKAN_VARIANT_TRACE="1")
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=args.timeout, env=environment)
                (out / (label + ".log")).write_text(completed.stdout + completed.stderr)
                record["exit"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError(f"native probe exited {completed.returncode}")
                report = json.loads(report_path.read_text())
                validate_residency(report, mode, target, getattr(args, "expected_device_id", None))
                if report["vocab_size"] != expected["vocab_size"]:
                    raise ValueError("vocabulary shape differs")
                actual = report.get("logits_bits")
                if actual is None:
                    actual = words(report["logits"])
                record.update(compare_logits(actual, expected["bits"], expected["visible"]))
                record.update({key: report[key] for key in ["device", "architecture", "streamed_linears", "source_first_parameters"]})
                record.update({key: report[key] for key in ["vendor_id", "device_id"] if key in report})
                for prefix, field in [("variant-plan:", "variant_plan"), ("variant-dispatch:", "variant_dispatches")]:
                    record[field] = {}
                    for line in completed.stderr.splitlines():
                        if line.startswith(prefix):
                            record[field].update(token.split("=", 1) for token in line.split()[1:])
                expected_family = {"gpt2_wide": "linear-bias-forward", "llama_wide": "linear-forward"}.get(case["family"])
                if expected_family and target == "gen9":
                    record["optimized_dispatch_verified"] = (
                        record["variant_plan"].get(expected_family) == "intel-gen9-lds"
                        and int(record["variant_dispatches"].get(expected_family, 0)) > 0)
                    record["pass"] = record["pass"] and record["optimized_dispatch_verified"]
                mode_words[mode] = actual
                if mode == "streaming" and "resident" in mode_words:
                    differences = sum(a != b for a, b in zip(actual, mode_words["resident"], strict=True))
                    record["resident_word_differences"] = differences
                    record["pass"] = record["pass"] and differences == 0
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                record["pass"] = False
                record["error"] = str(error)
            record["elapsed_seconds"] = time.monotonic() - started
            result["records"].append(record)
            print(f"{'PASS' if record['pass'] else 'FAIL'} {label}: "
                  f"{record.get('max_abs', record.get('error'))}", flush=True)
            write_json(out / "qualification.json", result)
    result["pass"] = bool(result["records"]) and all(record["pass"] for record in result["records"])
    tested = {record["family"] for record in result["records"]}
    result["missing_rdna_profiles"] = sorted(set(manifest["rdna_verified_profiles"]) - tested)
    result["rdna_profile_matrix_qualified"] = (
        set(args.modes) == {"resident", "streaming"}
        and not result["missing_rdna_profiles"]
        and all(record["pass"] for record in result["records"]
                if record["family"] in manifest["rdna_verified_profiles"]))
    result["native_architectures"] = sorted({record["architecture"] for record in result["records"] if "architecture" in record})
    result["unmeasured_registry_architectures"] = sorted(set(manifest["registry"]["canonical"]) - set(result["native_architectures"]))
    result["full_registry_qualified"] = (result["pass"]
        and set(args.modes) == {"resident", "streaming"}
        and not result["unmeasured_registry_architectures"])
    if digest(binary) != result["binary_sha256"]:
        raise ValueError("probe binary changed during qualification")
    write_json(out / "qualification.json", result)
    return 0 if result["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+")
    p = sub.add_parser("run")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--families", nargs="+")
    p.add_argument("--modes", nargs="+", choices=["resident", "streaming"], default=["resident", "streaming"])
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--target", choices=["gen9", "rdna3"], default="gen9")
    p.add_argument("--expected-device-id", type=lambda value: int(value, 0))
    args = parser.parse_args()
    if args.command == "run" and len(args.modes) != len(set(args.modes)):
        parser.error("residency modes must be unique")
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())

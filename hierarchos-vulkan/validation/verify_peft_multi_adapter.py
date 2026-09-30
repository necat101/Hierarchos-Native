from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

import torch

import verify_peft_lora_strict as strict


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = strict.FIXTURE_ROOT
BASE_DIR = FIXTURE_ROOT / "base"
FIXTURE_JSON = FIXTURE_ROOT / "fixture.json"
SAVED = os.environ.get("HIERARCHOS_PEFT_SAVED_MODULE")
MULTI_ROOT = FIXTURE_ROOT / ("multi-adapter" if not SAVED else "multi-saved-" + SAVED)
TOLERANCE = 2.0e-7

for dependency in [ROOT / ".k3-oracle-pydeps", ROOT / ".validation-pydeps"]:
    if dependency.is_dir(): sys.path.insert(0, str(dependency))
sys.path.insert(0, str(strict.TRANSFORMERS_ROOT / "src"))
_reference_class = None


def load_base():
    global _reference_class
    if _reference_class is None:
        _reference_class = type(strict._reference_model()[0])
    model = _reference_class.from_pretrained(BASE_DIR, attn_implementation="eager")
    model.config._attn_implementation = "eager"
    return model


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _max_abs(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"length mismatch: native={len(left)} reference={len(right)}")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _write_adapter(directory: Path, seed: int) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if directory.exists():
        shutil.rmtree(directory)
    if strict.FAMILY in {"gpt2", "llama", "mixtral", "qwen3_next"}:
        targets = (["c_attn"] if SAVED else ["c_proj"]) if strict.FAMILY == "gpt2" else ["v_proj"]
    else:
        from safetensors.torch import load_file
        weights = load_file(str(FIXTURE_ROOT / "adapter" / "adapter_model.safetensors"))
        targets = [key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
                   for key in weights if key.endswith(".lora_A.weight")]
        targets = [name for name in targets if not SAVED or (name != SAVED and not name.endswith("." + SAVED))]
        if not targets: raise RuntimeError("saved-module fixture has no independent LoRA target")
    model = get_peft_model(
        load_base(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=targets,
            modules_to_save=[SAVED] if SAVED else None,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name or ".modules_to_save." in name:
                parameter.normal_(mean=0.0, std=0.02)
    model.save_pretrained(directory, safe_serialization=True)


def _native_logits(binary: Path, adapter_a: Path, adapter_b: Path, selected: str, *, disabled: bool = False):
    output = MULTI_ROOT / f"native-{selected}{'-disabled' if disabled else ''}.json"
    args = [
        str(binary),
        "--model",
        str(BASE_DIR),
        "--fixture",
        str(FIXTURE_JSON),
        "--output",
        str(output),
        "--lora-adapter-named",
        "adapter_a",
        str(adapter_a),
        "--lora-adapter-named",
        "adapter_b",
        str(adapter_b),
        "--active-adapter",
        selected,
    ]
    if disabled:
        args.append("--disable-adapter")
    _run(args)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    strict.local_transformers_provenance()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not BASE_DIR.is_dir() or not FIXTURE_JSON.is_file():
        strict.main()
    MULTI_ROOT.mkdir(parents=True, exist_ok=True)
    adapter_a = MULTI_ROOT / "adapter-a"
    adapter_b = MULTI_ROOT / "adapter-b"
    _write_adapter(adapter_a, 101)
    _write_adapter(adapter_b, 202)

    fixture = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    input_ids = torch.tensor(fixture["input_ids"], dtype=torch.long).reshape(
        fixture["batch_size"], fixture["seq_len"]
    )
    attention_mask = torch.tensor(fixture["attention_mask"], dtype=torch.long).reshape(
        fixture["batch_size"], fixture["seq_len"]
    )

    reference_model = PeftModel.from_pretrained(
        load_base(), adapter_a, adapter_name="adapter_a"
    ).eval()
    reference_model.load_adapter(adapter_b, adapter_name="adapter_b")

    def reference(adapter_name: str) -> list[float]:
        reference_model.set_adapter(adapter_name)
        with torch.no_grad():
            return (
                reference_model(input_ids=input_ids, attention_mask=attention_mask)
                .logits.float()
                .flatten()
                .tolist()
            )

    hf_a = reference("adapter_a")
    hf_b = reference("adapter_b")
    reference_model.set_adapter("adapter_a")
    with reference_model.disable_adapter():
        with torch.no_grad():
            hf_disabled = (
                reference_model(input_ids=input_ids, attention_mask=attention_mask)
                .logits.float()
                .flatten()
                .tolist()
            )

    binary = strict._native_binary()
    native_a = _native_logits(binary, adapter_a, adapter_b, "adapter_a")
    native_b = _native_logits(binary, adapter_a, adapter_b, "adapter_b")
    native_disabled = _native_logits(binary, adapter_a, adapter_b, "adapter_a", disabled=True)

    report = {
        "peft_tolerance": TOLERANCE,
        "modules_to_save": SAVED,
        "adapter_a_max_abs": _max_abs(native_a["logits"], hf_a),
        "adapter_b_max_abs": _max_abs(native_b["logits"], hf_b),
        "disabled_base_max_abs": _max_abs(native_disabled["logits"], hf_disabled),
        "loaded_adapters": native_a["loaded_adapters"],
        "adapter_a_selected": native_a["active_adapter"] == "adapter_a" and native_a["adapters_enabled"],
        "adapter_b_selected": native_b["active_adapter"] == "adapter_b" and native_b["adapters_enabled"],
        "disabled_state": native_disabled["active_adapter"] == "adapter_a" and not native_disabled["adapters_enabled"],
    }
    report["pass"] = (
        report["adapter_a_max_abs"] <= TOLERANCE
        and report["adapter_b_max_abs"] <= TOLERANCE
        and report["disabled_base_max_abs"] <= TOLERANCE
        and report["loaded_adapters"] == ["adapter_a", "adapter_b"]
        and report["adapter_a_selected"]
        and report["adapter_b_selected"]
        and report["disabled_state"]
    )
    print(json.dumps(report, indent=2))
    (MULTI_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

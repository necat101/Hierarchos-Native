from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from verify_hf_logits import build_binary, tiny_models
from verify_hf_training import compare_training
from verify_qwen25_generation import run_family as verify_generation
from verify_qwen25_roundtrip import verify_family as verify_roundtrip


FAMILY = "gemma3"
FIXTURE_SEED = 0x48494552

_build_binary = build_binary


def build_binary(name: str) -> Path:
    if name == "transformer_" + "logits":
        name = "hierarchos-vulkan-transformer-logits"
    return _build_binary(name)


_verify_roundtrip = verify_roundtrip


def verify_roundtrip(*args, **kwargs):
    kwargs["expected_model_type"] = "gemma3_text"
    return _verify_roundtrip(*args, **kwargs)


def gemma3_model(*, training_reference: bool):
    torch.manual_seed(FIXTURE_SEED)
    matches = [
        model
        for name, model, _ in tiny_models(training_reference=training_reference)
        if name == FAMILY
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {FAMILY} fixture, found {len(matches)}")
    return matches[0]


def gemma3_tied_softcap_model(*, training_reference: bool):
    torch.manual_seed(FIXTURE_SEED)
    matches = [
        model
        for _, model, _ in tiny_models(training_reference=training_reference)
        if getattr(model.config, "model_type", None) == "gemma3_text"
        and bool(getattr(model.config, "tie_word_embeddings", False))
        and getattr(model.config, "attn_logit_softcapping", None) is not None
        and getattr(model.config, "final_logit_softcapping", None) is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one tied+softcapped Gemma 3 fixture, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Gemma 3 text cache, AdamW, and round-trip parity"
    )
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--generation-steps", type=int, default=3)
    args = parser.parse_args()
    if args.generation_steps <= 0:
        parser.error("--generation-steps must be positive")

    generation_binary = build_binary("transformer_generation_parity")
    training_binary = build_binary("transformer_parity")
    logits_binary = build_binary("transformer_logits")

    with tempfile.TemporaryDirectory(prefix="vulkan-gemma3-strict-") as directory:
        root = Path(directory)
        generation = verify_generation(
            generation_binary,
            root / "generation",
            FAMILY,
            gemma3_model(training_reference=False),
            args.generation_steps,
            args.device_index,
        )
        training = compare_training(
            training_binary,
            root / "training",
            FAMILY,
            gemma3_model(training_reference=True),
            args.device_index,
            False,
            2,
        )
        roundtrip = verify_roundtrip(
            training_binary,
            logits_binary,
            root / "roundtrip",
            FAMILY,
            gemma3_model(training_reference=True),
            args.device_index,
        )
        tied_softcap_generation = verify_generation(
            generation_binary,
            root / "tied_softcap_generation",
            FAMILY,
            gemma3_tied_softcap_model(training_reference=False),
            args.generation_steps,
            args.device_index,
        )
        tied_softcap_training = compare_training(
            training_binary,
            root / "tied_softcap_training",
            FAMILY,
            gemma3_tied_softcap_model(training_reference=True),
            args.device_index,
            False,
            2,
        )
        tied_softcap_roundtrip = verify_roundtrip(
            training_binary,
            logits_binary,
            root / "tied_softcap_roundtrip",
            FAMILY,
            gemma3_tied_softcap_model(training_reference=True),
            args.device_index,
        )

    print(json.dumps({
        "result": "pass",
        "family": FAMILY,
        "scope": "text-only",
        "generation": generation,
        "training": training,
        "roundtrip": roundtrip,
        "tied_softcap": {
            "generation": tied_softcap_generation,
            "training": tied_softcap_training,
            "roundtrip": tied_softcap_roundtrip,
        },
    }, indent=2))


if __name__ == "__main__":
    main()

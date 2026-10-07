"""Capture HF Gemma3 attention operands (post-rope Q/K, V, attention output).

Reuses the boundary-probe capture path so the operands are the oracle's own
tensors, then saves them for operand-matched replay. Diagnostic only.
"""
from __future__ import annotations

import json
import pathlib
import sys

# The oracle modules resolve their family and fixture root at import time.
sys.argv = ["diagnose_gemma3_rope_operands", "gemma3"]

import torch  # noqa: E402

import diagnose_peft_backward_boundary as boundary  # noqa: E402
import verify_peft_lora_strict as strict  # noqa: E402
import verify_peft_multi_adapter as multi  # noqa: E402

OUT = pathlib.Path(r"C:\Users\User\AppData\Local\Temp\hier_gemma3_hf_rope.pt")


def main() -> None:
    torch.set_num_threads(1)
    assert strict.FAMILY == "gemma3", strict.FAMILY
    multi.SAVED = "model.embed_tokens"
    adapter = pathlib.Path(strict.FIXTURE_ROOT) / "rope-probe-adapter"
    multi._write_adapter(adapter, 101)
    fixture = json.loads(multi.FIXTURE_JSON.read_text(encoding="utf-8"))
    ids = torch.tensor(fixture["input_ids"]).reshape(
        fixture["batch_size"], fixture["seq_len"]
    )
    mask = torch.tensor(fixture["attention_mask"]).reshape_as(ids)
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        multi.load_base(), adapter, is_trainable=True
    ).train()
    capture, rotary_calls, loss = boundary._capture(model, None, ids, mask)
    saved = {
        "capture": {key: value for key, value in capture.items() if torch.is_tensor(value)},
        "rotary": [
            {
                "call": index,
                "inputs": [tensor.clone() for tensor in call["inputs"]],
                "outputs": [entry["value"].clone() for entry in call["outputs"]],
            }
            for index, call in enumerate(rotary_calls)
        ],
        "loss": loss,
    }
    torch.save(saved, OUT)
    for entry in saved["rotary"]:
        print("rotary call", entry["call"],
              [(tuple(t.shape), float(t.abs().max())) for t in entry["outputs"]])
    print("saved:", OUT)


if __name__ == "__main__":
    main()

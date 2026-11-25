#!/usr/bin/env python3
"""Phase XIII per-projection route table for the production GQA LoRA fixture.

Every predicate below is transcribed from the source it mirrors, and the whole
table is checked against the dispatch counters the production graph actually
recorded in `raw/11_stability_lora_mixed.json` (LoRA) and
`raw/41_full_mixed_9.json` (full-parameter). If the predicted totals and the
measured counters disagree the script fails, so the table cannot claim a route
the graph did not take.

Source of each rule:
  * `wmma::wmma_shape_eligible`     src/wmma.rs
  * `wmma::WMMA_MIN_PROFITABLE_MACS`/`wmma_shape_profitable`  src/wmma.rs
  * `TransformerKernels::wmma_linear` executor admission      src/transformer.rs
  * `LinearWmma::record_forward/record_weight_grad/record_input_grad` admits()
                                                             src/wmma_linear.rs
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

LAYERS = 2
HIDDEN = 1024
INTERMEDIATE = 1024
VOCAB = 1024
HEADS_Q = 8
HEADS_KV = 2
HEAD_DIM = 128
BATCH = 2
SEQ = 64
ROWS = BATCH * SEQ
LORA_RANK = 8

WMMA_MIN_PROFITABLE_MACS = 16_000_000


def eligible(rows: int, columns: int, reduction: int) -> bool:
    return (
        rows >= 16
        and columns >= 16
        and reduction >= 16
        and rows * columns >= 1024
        and ((rows + 31) // 32) * ((columns + 31) // 32) >= 4
    )


def profitable(rows: int, columns: int, reduction: int) -> bool:
    return eligible(rows, columns, reduction) and rows * columns * reduction >= WMMA_MIN_PROFITABLE_MACS


def projector(name: str, kind: str, in_dim: int, out_dim: int, train_base: bool) -> dict:
    fwd = dict(rows=ROWS, columns=out_dim, reduction=in_dim)
    dx = dict(rows=ROWS, columns=in_dim, reduction=out_dim)
    dw = dict(rows=out_dim, columns=in_dim, reduction=ROWS)
    executor = profitable(fwd["rows"], fwd["columns"], fwd["reduction"])
    # `LinearWmma::admits` tests the projection geometry (batch rows, out, in)
    # for every phase; `select_arm` then applies the phase's own geometry.
    fwd_ok = executor and eligible(fwd["rows"], out_dim, in_dim) and eligible(fwd["rows"], out_dim, in_dim)
    dx_ok = executor and eligible(fwd["rows"], out_dim, in_dim) and eligible(dx["rows"], dx["columns"], dx["reduction"])
    dw_ok = (
        executor
        and eligible(fwd["rows"], out_dim, in_dim)
        and eligible(dw["rows"], dw["columns"], dw["reduction"])
        and train_base
    )
    reason = None
    if not eligible(fwd["rows"], out_dim, in_dim):
        reason = "wmma_shape_eligible: tile cannot be filled"
    elif not profitable(fwd["rows"], out_dim, in_dim):
        reason = (
            f"wmma_shape_profitable: {fwd['rows'] * out_dim * in_dim:,} MACs "
            f"< WMMA_MIN_PROFITABLE_MACS {WMMA_MIN_PROFITABLE_MACS:,}"
        )
    return {
        "projection": name,
        "kind": kind,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "forward": {**fwd, "macs": fwd["rows"] * fwd["columns"] * fwd["reduction"], "eligible": fwd_ok, "route": "wmma-mixed-f32acc" if fwd_ok else "fp32"},
        "input_grad": {**dx, "macs": dx["rows"] * dx["columns"] * dx["reduction"], "eligible": dx_ok, "route": "wmma-mixed-f32acc" if dx_ok else "fp32"},
        "weight_grad": {
            **dw,
            "macs": dw["rows"] * dw["columns"] * dw["reduction"],
            "eligible": dw_ok,
            "route": "wmma-mixed-f32acc" if dw_ok else ("not-recorded (base frozen)" if not train_base else "fp32"),
        },
        "train_base": train_base,
        "fallback_reason": reason,
    }


def graph(train_base: bool) -> list[dict]:
    rows = []
    for layer in range(LAYERS):
        rows += [
            projector(f"model.layers.{layer}.self_attn.q_proj", "dense projection", HIDDEN, HEADS_Q * HEAD_DIM, train_base),
            projector(f"model.layers.{layer}.self_attn.k_proj", "dense projection", HIDDEN, HEADS_KV * HEAD_DIM, train_base),
            projector(f"model.layers.{layer}.self_attn.v_proj", "dense projection", HIDDEN, HEADS_KV * HEAD_DIM, train_base),
            projector(f"model.layers.{layer}.self_attn.o_proj", "dense projection", HEADS_Q * HEAD_DIM, HIDDEN, train_base),
            projector(f"model.layers.{layer}.mlp.gate_proj", "dense projection", HIDDEN, INTERMEDIATE, train_base),
            projector(f"model.layers.{layer}.mlp.up_proj", "dense projection", HIDDEN, INTERMEDIATE, train_base),
            projector(f"model.layers.{layer}.mlp.down_proj", "dense projection", INTERMEDIATE, HIDDEN, train_base),
        ]
    rows.append(projector("lm_head", "vocabulary projection", HIDDEN, VOCAB, train_base))
    return rows


def adapter_rows() -> list[dict]:
    rows = []
    for layer in range(LAYERS):
        rows.append(
            {
                "projection": f"model.layers.{layer}.self_attn.q_proj.lora_A",
                "kind": "LoRA adapter A",
                "in_dim": HIDDEN,
                "out_dim": LORA_RANK,
                "forward": {"rows": ROWS, "columns": LORA_RANK, "reduction": HIDDEN,
                            "macs": ROWS * LORA_RANK * HIDDEN, "eligible": False, "route": "fp32 narrow-output lane4"},
                "input_grad": {"rows": ROWS, "columns": HIDDEN, "reduction": LORA_RANK,
                               "macs": ROWS * HIDDEN * LORA_RANK, "eligible": False, "route": "fp32 group4-muladd"},
                "weight_grad": {"rows": LORA_RANK, "columns": HIDDEN, "reduction": ROWS,
                                "macs": LORA_RANK * HIDDEN * ROWS, "eligible": False, "route": "fp32 linear_weight_grad"},
                "train_base": True,
                "fallback_reason": "wmma_shape_eligible: out_dim 8 < 16 and 1.05 MMAC < 16 MMAC floor",
            }
        )
        rows.append(
            {
                "projection": f"model.layers.{layer}.self_attn.q_proj.lora_B",
                "kind": "LoRA adapter B",
                "in_dim": LORA_RANK,
                "out_dim": HIDDEN,
                "forward": {"rows": ROWS, "columns": HIDDEN, "reduction": LORA_RANK,
                            "macs": ROWS * HIDDEN * LORA_RANK, "eligible": False, "route": "fp32 linear_forward"},
                "input_grad": {"rows": ROWS, "columns": LORA_RANK, "reduction": HIDDEN,
                               "macs": ROWS * LORA_RANK * HIDDEN, "eligible": False, "route": "fp32 linear_input_grad"},
                "weight_grad": {"rows": HIDDEN, "columns": LORA_RANK, "reduction": ROWS,
                                "macs": HIDDEN * LORA_RANK * ROWS, "eligible": False, "route": "fp32 linear_weight_grad"},
                "train_base": True,
                "fallback_reason": "wmma_shape_eligible: reduction 8 < 16 (LoRA rank)",
            }
        )
    return rows


def counts(rows: list[dict], phase: str) -> int:
    return sum(1 for row in rows if row[phase]["eligible"])


def main() -> int:
    lora = report_counts("11_stability_lora_mixed.json")
    full = report_counts("41_full_mixed_9.json")
    lora_graph = graph(train_base=False)
    full_graph = graph(train_base=True)
    adapters = adapter_rows()

    prediction = {
        "lora": {
            "forward": counts(lora_graph, "forward"),
            "weight_grad": counts(lora_graph, "weight_grad"),
            "input_grad": counts(lora_graph, "input_grad"),
        },
        "full": {
            "forward": counts(full_graph, "forward"),
            "weight_grad": counts(full_graph, "weight_grad"),
            "input_grad": counts(full_graph, "input_grad"),
        },
    }
    measured = {
        "lora": {
            "forward": lora["wmma_forward"],
            "weight_grad": lora["wmma_weight_grad"],
            "input_grad": lora["wmma_input_grad"],
        },
        "full": {
            "forward": full["wmma_forward"],
            "weight_grad": full["wmma_weight_grad"],
            "input_grad": full["wmma_input_grad"],
        },
        "steps": {"lora": lora["steps"], "full": full["steps"]},
    }
    for mode in prediction:
        for phase in prediction[mode]:
            if prediction[mode][phase] != measured[mode][phase]:
                raise SystemExit(
                    f"{mode}/{phase}: predicted {prediction[mode][phase]} cooperative dispatches, "
                    f"measured {measured[mode][phase]}"
                )
    out = {
        "geometry": {
            "layers": LAYERS, "hidden": HIDDEN, "intermediate": INTERMEDIATE, "vocab": VOCAB,
            "q_heads": HEADS_Q, "kv_heads": HEADS_KV, "head_dim": HEAD_DIM,
            "batch": BATCH, "seq": SEQ, "gemm_rows": ROWS, "lora_rank": LORA_RANK,
        },
        "lora": lora_graph + adapters,
        "full": full_graph,
        "predicted": prediction,
        "measured": measured,
        "shape_fallbacks": {"lora": lora["wmma_shape_fallbacks"], "full": full["wmma_shape_fallbacks"]},
        "macs_per_step": {
            "forward": sum(row["forward"]["macs"] for row in lora_graph),
            "input_grad": sum(row["input_grad"]["macs"] for row in lora_graph),
            "weight_grad": sum(row["weight_grad"]["macs"] for row in full_graph),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def report_counts(name: str) -> dict:
    """Per-step counters: the harness reports whole-run totals."""
    report = json.loads((RAW / name).read_text())
    steps = len(report["records"])
    out = {"steps": steps}
    for key in ("wmma_forward", "wmma_weight_grad", "wmma_input_grad", "wmma_shape_fallbacks"):
        if report[key] % steps:
            raise SystemExit(f"{name}: {key}={report[key]} is not a whole number of steps")
        out[key] = report[key] // steps
    return out


if __name__ == "__main__":
    raise SystemExit(main())

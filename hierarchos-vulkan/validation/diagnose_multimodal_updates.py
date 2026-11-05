"""Separate backbone/projector update drift using frozen CPU step-two states.

Each fork uses the probe's first forward before its extra optimizer steps.
This is diagnostic evidence, never a replacement for the original strict gate.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess

from qualify_multimodal_training import compare, digest, read_f32_tensors, write_json


def write_f32_tensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (shape, values) in sorted(tensors.items()):
        start = len(payload)
        payload.extend(struct.pack(f"<{len(values)}f", *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--after-step", type=int, choices=[1, 2], default=2)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, native = args.fixtures / args.case, args.native / args.case
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cpu = json.loads((source / "cpu.json").read_text())
    actual = json.loads((native / "report.json").read_text())
    fixture = json.loads((source / "input.json").read_text())
    cpu_model = out / "cpu-model"
    cpu_model.mkdir()
    shutil.copyfile(source / "model" / "config.json", cpu_model / "config.json")
    tensors = read_f32_tensors(source / "model" / "model.safetensors")
    updated = read_f32_tensors(source / f"cpu-step-{args.after_step}.safetensors")
    tensors.update(updated)
    for alias, canonical in cpu["aliases"].items():
        if alias in tensors and canonical in updated:
            tensors[alias] = updated[canonical]
    write_f32_tensors(cpu_model / "model.safetensors", tensors)
    records = []
    for model_kind, model in [("cpu", cpu_model), ("native", native / f"step-{args.after_step}")]:
        for projector_kind, trajectory in [("cpu", cpu), ("native", actual)]:
            name = f"{model_kind}-backbone-{projector_kind}-projector"
            state = trajectory["steps"][args.after_step - 1]
            fork = {**fixture, "projector_weight": state["projector_weight"],
                "projector_bias": state["projector_bias"]}
            path = out / f"{name}.json"
            write_json(path, fork)
            process = subprocess.run([str(args.binary.resolve()), "--model", str(model.resolve()),
                "--fixture", str(path), "--output", str(out / name)], capture_output=True, text=True)
            (out / f"{name}.log").write_text(process.stdout + process.stderr)
            process.check_returncode()
            report = json.loads((out / name / "report.json").read_text())
            visible = cpu["visible"]
            logits = report["steps"][0]["logits"]
            reference = cpu["final_logits"] if args.after_step == 2 else cpu["steps"][1]["logits"]
            if len(logits) != len(visible) or len(reference) != len(visible):
                raise ValueError("diagnostic logit shape differs")
            metrics = compare([x for x, keep in zip(logits, visible) if keep],
                [x for x, keep in zip(reference, visible) if keep])
            record = {"backbone": model_kind, "projector": projector_kind, "logits": metrics}
            if args.after_step == 1:
                for key in ["modality_embedding_grad", "feature_grad"]:
                    record[key] = compare(report["steps"][0][key], cpu["steps"][1][key])
            records.append(record)
            print(name, record, flush=True)
    write_json(out / "diagnosis.json", {"scope": "update_drift_attribution_only", "case": args.case,
        "after_step": args.after_step,
        "cpu_sha256": digest(source / "cpu.json"), "native_sha256": digest(native / "report.json"),
        "binary_sha256": digest(args.binary), "runner_sha256": digest(__file__), "records": records})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the real two-rank Windows/Linux TCP trainer and compare optimizer bytes.

Both ranks use this machine's adapter (loopback transport qualification). This
does not certify two physical GPUs, remote Ethernet performance, or native RDMA.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CRATE = Path(__file__).resolve().parents[1]


def build(profile: str) -> dict[str, Path]:
    command = ["cargo", "build", "-j", "1", "--bin", "training_step_probe", "--bin",
               "hierarchos-vulkan-train", "--message-format=json-render-diagnostics"]
    if profile == "release":
        command.append("--release")
    result = subprocess.run(command, cwd=CRATE, capture_output=True, text=True, check=True)
    binaries = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        if row.get("reason") == "compiler-artifact" and row.get("executable"):
            binaries[row["target"]["name"]] = Path(row["executable"])
    return binaries


def tensor_bytes(path: Path) -> dict:
    from safetensors import safe_open
    tensors = {}
    for filename in ("model.safetensors", "optimizer.safetensors"):
        with safe_open(path / filename, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                value = archive.get_tensor(key)
                tensors[f"{filename}/{key}"] = (str(value.dtype), tuple(value.shape), value.numpy().tobytes())
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("release", "debug"), default="debug")
    parser.add_argument("--output", type=Path, default=CRATE / "target" / "network-training-parity")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=out))
    binaries = build(args.profile)
    model = work / "model"
    subprocess.run([str(binaries["training_step_probe"]), "--model-dir", str(model),
                    "--hidden", "32", "--context-dim", "32", "--vocab", "32", "--batch", "1",
                    "--tokens", "2", "--steps", "1", "--warmup", "0"], cwd=CRATE,
                   capture_output=True, text=True, check=True)
    dataset = work / "data.jsonl"
    dataset.write_text("\n".join(json.dumps({"input_ids": row})
                                   for row in ([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])) + "\n")
    common = [str(binaries["hierarchos-vulkan-train"]), "--model", str(model), "--dataset", str(dataset),
              "--epochs", "1", "--batch-size", "1", "--gradient-accumulation-steps", "2",
              "--no-shuffle", "--precision", "fp32", "--grad-clip", "0",
              "--gradient-stream-chunk-values", "257"]
    expected = None
    rows = []
    for mode in ("param-server", "allreduce"):
        processes, logs = [], []
        try:
            for rank in (0, 1):
                log = (work / f"{mode}-rank{rank}.log").open("w")
                logs.append(log)
                command = [*common, "--output", str(work / f"{mode}-rank{rank}"), "--world-size", "2",
                           "--rank", str(rank), "--peer-ip", "127.0.0.1", "--network-transport", "tcp",
                           "--distributed-mode", mode]
                environment = dict(os.environ, HIERARCHOS_NETWORK_TIMEOUT_SECONDS="120")
                processes.append(subprocess.Popen(command, cwd=CRATE, stdout=log,
                                                  stderr=subprocess.STDOUT, env=environment))
            for process in processes:
                if process.wait(timeout=900):
                    raise RuntimeError(f"{mode} trainer failed; see {work}")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
        actual = tensor_bytes(work / f"{mode}-rank0")
        if expected is None:
            expected = actual
        differences = sorted(key for key in expected.keys() | actual.keys()
                             if expected.get(key) != actual.get(key))
        rows.append({"mode": mode, "reference": "param-server", "tensors": len(actual), "differences": differences,
                     "pass": not differences})
        print(f"{mode}: {len(actual)} model/optimizer tensors, {len(differences)} differences", flush=True)
    result = "pass" if all(row["pass"] for row in rows) else "fail"
    (out / "receipt.json").write_text(json.dumps({"result": result, "scope": "two processes, one physical adapter, TCP loopback",
                                                  "work": str(work), "rows": rows}, indent=2) + "\n")
    return int(result != "pass")


if __name__ == "__main__":
    sys.exit(main())

"""Pin AVX2 CPU cross entropy on unchanged training-fixture logits and targets.

This deliberately regenerates only the host-dependent loss operator. Model
outputs, gradients, factors, moments and the original fixtures stay frozen.
Each scalar is bound to every input word by a SHA-256 digest.
"""
import hashlib
import json
import struct
from pathlib import Path

import fixture_toolchain
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent / "fixtures"
FIXTURE = ROOT / "deepseek_v41_training_loss.avx2.json"


def record(records, logits, targets, vocab, source):
    normalized = [value if value >= 0 else 0xffffffff for value in targets]
    words = [len(targets), vocab, *logits, *normalized]
    digest = hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest()
    x = torch.tensor(logits, dtype=torch.uint32).view(torch.float32).reshape(-1, vocab)
    labels = torch.tensor([value if value >= 0 else -100 for value in targets])
    loss = F.cross_entropy(x, labels, ignore_index=-100)
    entry = {"loss_bits": loss.view(torch.uint32).item()}
    if digest in records and records[digest] != entry:
        raise AssertionError(f"same CPU input has conflicting losses: {source}")
    records[digest] = entry


def main():
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError("use the pinned AVX2 CPU for this sibling")
    fixture_toolchain.check_file(FIXTURE)
    torch.set_num_threads(1)
    records, sources = {}, {}
    names = ["projection_grad", "full_training", "full_mixed_training",
             "full_bf16_training_v3", "full_bf16_training_v3.avx2",
             "production_ratio_training_v2"]
    for name in names:
        path = ROOT / f"deepseek_v41_{name}.json"
        fixture_toolchain.check_file(path)
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixture = json.loads(path.read_text())
        if name == "projection_grad":
            for case in fixture["training"]:
                for step in case["steps"]:
                    record(records, step["logits_bits"], case["targets"], case["vocab"], path.name)
        else:
            for case in fixture["cases"]:
                targets = [value for batch in case["batches"] for value in batch["targets"]]
                for step in case["steps"]:
                    logits = [value for batch in step["logits_bits"] for value in batch]
                    record(records, logits, targets, len(logits) // len(targets), path.name)
    result = {"contract": "CPU cross entropy on immutable training logits and targets",
              "sources": sources, "records": records}
    fixture_toolchain.write_fixture(FIXTURE, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS {len(records)} CPU losses bound to complete immutable inputs")


if __name__ == "__main__":
    main()

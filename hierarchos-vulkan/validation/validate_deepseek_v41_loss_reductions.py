"""Replay CPU cross entropy on immutable, pre-existing logit words.

The AVX2 sibling records the host's eight-lane log-softmax reduction. Inputs
come only from the original frozen fixture; native outputs are never consumed.
The original fixture and its sixteen-lane expectations remain unchanged.
"""
import argparse
import json
from pathlib import Path

import fixture_toolchain
import torch
import torch.nn.functional as F

BASE = Path(__file__).parent / "fixtures/deepseek_v41_loss_reductions.json"
FIXTURE = fixture_toolchain.variant_path(BASE)
fixture_toolchain.check_file(BASE)
fixture_toolchain.check_file(FIXTURE)


def bits(value):
    return value.contiguous().reshape(-1).view(torch.uint32).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    frozen = json.loads(BASE.read_text())
    cases = []
    for case in frozen["cases"]:
        logits = torch.tensor(case["logits_bits"], dtype=torch.uint32).view(torch.float32)
        logits = logits.reshape(case["rows"], case["vocab"])
        targets = torch.tensor(case["targets"])
        rows = F.cross_entropy(logits, targets, reduction="none")
        cases.append({**case, "row_loss_bits": bits(rows),
                      "row_mean_bits": bits(rows.mean()),
                      "reduced_mean_bits": bits(F.cross_entropy(logits, targets))})
    result = {**frozen, "cases": cases}
    if args.write_fixture:
        fixture_toolchain.write_fixture(FIXTURE, json.dumps(result, indent=2) + "\n")
    elif json.loads(FIXTURE.read_text()) != result:
        raise AssertionError("loss reduction fixture differs from CPU PyTorch")
    print(f"PASS {len(cases)} immutable-logit CPU cross-entropy cases")


if __name__ == "__main__":
    main()

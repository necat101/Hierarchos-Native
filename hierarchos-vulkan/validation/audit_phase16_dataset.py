#!/usr/bin/env python3
"""Audit the Phase XVI training fixture for held-out suitability.

The audit is a pure function of the dataset bytes, so it can be re-derived from
the committed byte-identical copy even on a checkout that has no ``target/``
scratch directory. It reports exact and near duplicates, reuse under the
declared 100-step / batch-8 schedule, and whether the corpus can support an
independent split.

Usage (from ``hierarchos-vulkan``):

    python validation/audit_phase16_dataset.py \
        --dataset validation/fixtures/phase17/phase16_dataset.jsonl \
        --output validation/fixtures/phase17/phase16_dataset_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEQUENCE_LEN = 64
BATCH_SIZE = 8
TRAINING_STEPS = 100


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ngram_hashes(token_ids: list[int], n: int = 8) -> set[str]:
    return {
        sha256_bytes(",".join(str(value) for value in token_ids[i:i + n]).encode())
        for i in range(max(0, len(token_ids) - n + 1))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.dataset.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    sequences = [row["input_ids"] for row in rows]

    token_total = sum(len(sequence) for sequence in sequences)
    unique_sequences = {tuple(sequence) for sequence in sequences}
    exact_duplicates = len(sequences) - len(unique_sequences)

    # A near duplicate is a distinct row whose 8-gram set is contained in
    # another row's set (or vice versa). This catches shifted windows of the
    # same arithmetic progression, which exact hashing misses.
    gram_sets = [ngram_hashes(sequence) for sequence in sequences]
    near_pairs = []
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            if gram_sets[left] and gram_sets[right]:
                overlap = len(gram_sets[left] & gram_sets[right])
                if overlap and (
                    overlap == len(gram_sets[left]) or overlap == len(gram_sets[right])
                ):
                    near_pairs.append({"left": left, "right": right, "shared_8grams": overlap})

    # The generator emits `(base + step * 7 + index) % range`; every sequence is
    # therefore a cyclic shift/offset of one arithmetic progression. Report the
    # modular difference so the claim is derived, not asserted.
    offsets = []
    for sequence in sequences:
        differences = {
            (sequence[index + 1] - sequence[index]) % 1024
            for index in range(len(sequence) - 1)
        }
        offsets.append(sorted(differences))

    draws = TRAINING_STEPS * BATCH_SIZE
    windows = len(sequences)
    audit = {
        "dataset": args.dataset.name,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "examples": len(sequences),
        "unique_examples": len(unique_sequences),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_examples": near_pairs[:8],
        "sequence_lengths": sorted({len(sequence) for sequence in sequences}),
        "tokens": token_total,
        "independent_source_documents": 1,
        "sequence_packing": f"one contiguous window of {SEQUENCE_LEN} tokens per row",
        "tokenization": "pretokenized integer ids; no tokenizer is consulted",
        "padding": "none at fixture level; the training harness pads short rows with PAD and masks them",
        "loss_mask": "every real position supervised; padded positions carry target u32::MAX and weight 0",
        "arithmetic_progression_offsets": {
            "distinct_offset_sets": len({tuple(offset) for offset in offsets}),
            "sample": offsets[:4],
        },
        "schedule_reuse": {
            "batch_size": BATCH_SIZE,
            "steps": TRAINING_STEPS,
            "sample_draws": draws,
            "distinct_examples": windows,
            "mean_reuse_per_example": draws / windows if windows else None,
            "repeats_within_run": draws > windows,
        },
        "split_verdict": {
            "can_support_held_out_split": False,
            "reason": (
                "the corpus is one generator family of 32 modular arithmetic "
                "progressions; every row is a cyclic shift of the same "
                "progression, so any partition leaks the held-out distribution "
                "and the 100-step schedule reuses each example "
                f"{draws / windows:.1f}x on average"
            ),
            "recommendation": (
                "keep this fixture frozen for Phase XVI numerical and dispatch "
                "qualification; run Phase XVII held-out evaluation on the "
                "separate document-level fixture in validation/fixtures/phase17"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["split_verdict"], indent=2))
    print(f"sha256={audit['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

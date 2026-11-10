#!/usr/bin/env python3
"""Deterministic multi-document held-out evaluation fixture (Phase XVII).

The Phase XVI fixture is a single synthetic generator family: 32 sequences of
modular arithmetic progressions. That is sufficient to exercise cooperative
dispatches and precision policies, but it cannot support a defensible
train/validation/test split, so Phase XVII needs a separate, versioned corpus.

This generator builds that corpus with standard-library Python only, so a
checkout without numpy can still reproduce every committed byte. Every choice
is seeded and every ordering is derived from SHA-256, which means the
train/validation/test manifests, token ids and vocabulary are a pure function
of this file.

Design:
  * Each *document* is generated from its own seed and its own family.
  * Documents are independent: they share a grammar and a vocabulary but no
    line, window or entity pairing recurs across documents.
  * Splitting happens at document granularity, before tokenization sees a
    held-out document. A vocabulary is built from training documents only, so
    validation/test token identity cannot leak into the optimizer.
  * Windows are cut per document and never cross document boundaries.

Usage (from ``hierarchos-vulkan``):

    python validation/make_phase17_fixture.py --output validation/fixtures/phase17

Writes ``corpus.jsonl``, ``vocab.json``, ``train.jsonl``, ``validation.jsonl``,
``test.jsonl`` and ``split_manifest.json``. All are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FIXTURE_VERSION = "phase17-heldout-v1"
TOKENIZER_NAME = "hierarchos-word-hash-v1"
TOKENIZER_REVISION = "1"
SPLIT_SEED = 20261110
DOCUMENT_SEED = 20261111
DOCUMENTS = 320
LINES_PER_DOCUMENT = 40
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_COUNT = 4
VOCAB_LIMIT = 1024 - SPECIAL_COUNT

# Shared, family-scoped vocabularies. A document draws from its family's pools
# only, so the grammar is learnable while every combination stays fresh.
FAMILIES = {
    "catalog": {
        "items": [f"item{i:02d}" for i in range(60)],
        "colors": [f"color{i:02d}" for i in range(24)],
        "sizes": [f"size{i:02d}" for i in range(12)],
        "prices": [f"price{i:03d}" for i in range(120)],
    },
    "schedule": {
        "days": [f"day{i}" for i in range(7)],
        "times": [f"time{i:02d}" for i in range(24)],
        "people": [f"person{i:02d}" for i in range(40)],
        "verbs": [f"verb{i:02d}" for i in range(16)],
        "rooms": [f"room{i:02d}" for i in range(20)],
    },
    "log": {
        "levels": [f"level{i}" for i in range(6)],
        "modules": [f"module{i:02d}" for i in range(24)],
        "codes": [f"code{i:03d}" for i in range(160)],
        "attempts": [f"attempt{i}" for i in range(50)],
    },
}

GRAMMAR = {
    "catalog": [
        "items", "colors", "sizes", "prices", "items", "colors", "sizes",
        "prices", "usd", "items", "colors", "sizes",
    ],
    "schedule": [
        "days", "times", "people", "verbs", "people", "at", "rooms",
        "days", "times", "people", "verbs", "at", "rooms",
    ],
    "log": [
        "levels", "modules", "reported", "codes", "after", "attempts",
        "tries", "levels", "modules", "reported", "codes", "after",
    ],
}

FAMILY_ORDER = ("catalog", "schedule", "log")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rng(seed_material: str) -> "Rng":
    """A tiny SplitMix64 generator: identical on every Python version/platform."""
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return Rng(int.from_bytes(digest[:8], "big"))


class Rng:
    def __init__(self, state: int) -> None:
        self.state = state & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        return self.next_u64() % bound

    def choice(self, values: list[str]) -> str:
        return values[self.below(len(values))]


def generate_document(index: int, attempt: int = 0) -> dict:
    family = FAMILY_ORDER[index % len(FAMILY_ORDER)]
    rng = stable_rng(f"{DOCUMENT_SEED}:{index}:{family}:{attempt}")
    pools = FAMILIES[family]
    lines: list[list[str]] = []
    for _ in range(LINES_PER_DOCUMENT):
        words: list[str] = []
        for slot in GRAMMAR[family]:
            if slot in pools:
                words.append(rng.choice(pools[slot]))
            else:
                words.append(slot)
        lines.append(words)
    document_id = f"doc-{index:04d}-{family}"
    return {
        "document_id": document_id,
        "family": family,
        "lines": lines,
    }


def split_by_document(documents: list[dict]) -> dict[str, list[dict]]:
    """Document-level split ordered by a seeded hash of the document id.

    Sorting by hash rather than by index keeps the assignment independent of
    corpus construction order and of the family cycle.
    """
    ranked = sorted(documents, key=lambda d: sha256_text(f"{SPLIT_SEED}:{d['document_id']}"))
    total = len(ranked)
    train_count = int(math.floor(total * TRAIN_FRACTION))
    validation_count = int(math.floor(total * VALIDATION_FRACTION))
    if total - train_count - validation_count <= 0:
        raise SystemExit("corpus is too small for a non-empty test partition")
    return {
        "train": ranked[:train_count],
        "validation": ranked[train_count:train_count + validation_count],
        "test": ranked[train_count + validation_count:],
    }


def training_vocabulary(train_documents: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for document in train_documents:
        for line in document["lines"]:
            for word in line:
                counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    # The families share pools, so the realised vocabulary is expected to be
    # smaller than the model's 1024-entry embedding. The budget is a cap, not a
    # target: the committed vocab.json records what was actually built.
    return ordered[:VOCAB_LIMIT]


def tokenize(document: dict, word_to_id: dict[str, int]) -> list[int]:
    ids = [BOS]
    for line in document["lines"]:
        ids.extend(word_to_id.get(word, UNK) for word in line)
        ids.append(EOS)
    return ids


def build_windows(token_ids: list[int], seq_len: int) -> list[dict]:
    """One window per (document, start offset), never crossing a document.

    ``predict`` excludes the last token because no target exists for it; a
    short final window is padded with ``PAD`` and carries an explicit
    ``valid_tokens`` count rather than pretending the padding is supervised.
    """
    predict = len(token_ids) - 1
    if predict < 1:
        return []
    windows = []
    for start in range(0, predict, seq_len):
        count = min(predict - start, seq_len)
        ids = token_ids[start:start + count]
        targets = token_ids[start + 1:start + 1 + count]
        windows.append(
            {
                "window_index": len(windows),
                "start": start,
                "valid_tokens": count,
                "input_ids": ids,
                "targets": targets,
            }
        )
    return windows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def ngram_hashes(token_ids: list[int], n: int = 8) -> set[str]:
    return {
        sha256_text(",".join(str(value) for value in token_ids[i:i + n]))
        for i in range(max(0, len(token_ids) - n + 1))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    documents = [generate_document(index) for index in range(DOCUMENTS)]
    partitions = split_by_document(documents)

    # Leakage prevention by construction: a held-out document is regenerated
    # (same document id, next deterministic attempt) until none of its 8-grams
    # occurs in any other partition. Held-out identity stays fixed because the
    # split is a function of the document id, never of the document content.
    index_of = {document["document_id"]: index for index, document in enumerate(documents)}
    token_index = training_vocabulary(partitions["train"])
    provisional = {word: position + SPECIAL_COUNT for position, word in enumerate(token_index)}
    forbidden = {
        name: set() for name in ("train", "validation", "test")
    }
    for document in partitions["train"]:
        forbidden["train"] |= ngram_hashes(tokenize(document, provisional))
    for name in ("validation", "test"):
        for position, document in enumerate(partitions[name]):
            attempt = 0
            while True:
                candidate = generate_document(index_of[document["document_id"]], attempt)
                grams = ngram_hashes(tokenize(candidate, provisional))
                other = "test" if name == "validation" else "validation"
                if (
                    not grams & forbidden["train"]
                    and not grams & forbidden[name]
                    and not grams & forbidden[other]
                ):
                    partitions[name][position] = candidate
                    forbidden[name] |= grams
                    break
                attempt += 1
                if attempt > 512:
                    raise SystemExit(
                        f"could not make {document['document_id']} 8-gram disjoint from the "
                        "other partitions"
                    )
    documents = [
        next(
            document
            for partition in ("train", "validation", "test")
            for document in partitions[partition]
            if document["document_id"] == reference["document_id"]
        )
        for reference in documents
    ]
    vocabulary = training_vocabulary(partitions["train"])
    word_to_id = {word: index + SPECIAL_COUNT for index, word in enumerate(vocabulary)}

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    corpus_rows = [
        {
            "document_id": document["document_id"],
            "family": document["family"],
            "lines": document["lines"],
        }
        for document in documents
    ]
    write_jsonl(output / "corpus.jsonl", corpus_rows)

    vocab_payload = {
        "tokenizer": TOKENIZER_NAME,
        "revision": TOKENIZER_REVISION,
        "special_tokens": {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK},
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary) + SPECIAL_COUNT,
    }
    (output / "vocab.json").write_text(
        json.dumps(vocab_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    manifests: dict[str, dict] = {}
    partition_windows: dict[str, list[dict]] = {}
    partition_ngrams: dict[str, set[str]] = {}
    for name, members in partitions.items():
        rows: list[dict] = []
        windows: list[dict] = []
        ngrams: set[str] = set()
        tokens = 0
        for document in members:
            token_ids = tokenize(document, word_to_id)
            tokens += len(token_ids)
            ngrams |= ngram_hashes(token_ids)
            rows.append(
                {
                    "document_id": document["document_id"],
                    "family": document["family"],
                    "token_count": len(token_ids),
                    "input_ids": token_ids,
                }
            )
            for window in build_windows(token_ids, args.seq_len):
                windows.append(
                    {
                        "document_id": document["document_id"],
                        "family": document["family"],
                        **window,
                    }
                )
        write_jsonl(output / f"{name}.jsonl", rows)
        partition_windows[name] = windows
        partition_ngrams[name] = ngrams
        manifests[name] = {
            "documents": len(members),
            "documents_list": [document["document_id"] for document in members],
            "tokens": tokens,
            "windows": len(windows),
            "valid_target_tokens": sum(window["valid_tokens"] for window in windows),
            "families": {
                family: sum(1 for document in members if document["family"] == family)
                for family in FAMILY_ORDER
            },
            "sha256": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest(),
        }

    overlaps = {}
    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left >= right:
                continue
            shared = partition_ngrams[left] & partition_ngrams[right]
            overlaps[f"{left}__{right}"] = {
                "shared_8gram_hashes": len(shared),
                "shared_documents": len(
                    {document["document_id"] for document in partitions[left]}
                    & {document["document_id"] for document in partitions[right]}
                ),
            }
    if any(entry["shared_8gram_hashes"] for entry in overlaps.values()):
        raise SystemExit(f"8-gram overlap detected across partitions: {overlaps}")

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "generator": "validation/make_phase17_fixture.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seq_len": args.seq_len,
        "split_seed": SPLIT_SEED,
        "document_seed": DOCUMENT_SEED,
        "split_algorithm": (
            "documents sorted by sha256(f'{split_seed}:{document_id}'); "
            "first 80% train, next 10% validation, remaining 10% test"
        ),
        "vocabulary_source": "training partition only",
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "revision": TOKENIZER_REVISION,
            "vocabulary_size": len(vocabulary) + SPECIAL_COUNT,
            "vocab_sha256": hashlib.sha256((output / "vocab.json").read_bytes()).hexdigest(),
        },
        "corpus": {
            "documents": len(documents),
            "lines_per_document": LINES_PER_DOCUMENT,
            "families": list(FAMILY_ORDER),
            "sha256": hashlib.sha256((output / "corpus.jsonl").read_bytes()).hexdigest(),
        },
        "partitions": manifests,
        "overlap": overlaps,
        "notes": [
            "synthetic grammar corpus: generalization claims are scoped to these families",
            "no held-out document contributes tokens to the training vocabulary",
            "windows never cross a document boundary",
        ],
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "documents": len(documents),
                "vocabulary_size": len(vocabulary) + SPECIAL_COUNT,
                "partitions": {
                    name: {
                        "documents": entry["documents"],
                        "tokens": entry["tokens"],
                        "windows": entry["windows"],
                    }
                    for name, entry in manifests.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

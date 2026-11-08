#!/usr/bin/env python3
"""Select the one test executable from Cargo JSON compiler messages.

`cargo test --message-format=json` is the deterministic way to identify the
linked test harness. Directory scans by modification time can select an
unrelated stale executable, especially on shared self-hosted runners.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def select(path: pathlib.Path, target: str) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        artifact_target = message.get("target", {})
        executable = message.get("executable")
        if executable and artifact_target.get("name") == target:
            candidates.append(pathlib.Path(executable).resolve())
    if len(candidates) != 1:
        raise ValueError(f"expected one {target} test executable, got {candidates}")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", type=pathlib.Path)
    parser.add_argument("--target", default="hierarchos_vulkan")
    args = parser.parse_args(argv)
    try:
        print(select(args.messages, args.target).as_posix())
    except (OSError, ValueError) as error:
        print(f"cargo_test_binary: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regression tests for the Windows-safe Safetensors replacement helper.

The packed-cell fixture builder briefly failed on Windows with os error 1224
because `save_file` truncated a path while a tensor from that file still owned a
memory mapping.  These tests exercise the two supported workflows: writing the
package twice to the same directory, and replacing a package after loading and
releasing its tensors.
"""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

import verify_rwkv_cpu_bridge as bridge


class AtomicSafetensorsReplaceTest(unittest.TestCase):
    def test_writing_a_package_twice_replaces_the_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safetensors-atomic-") as scratch:
            directory = Path(scratch)
            bridge.write_package(directory)
            path = directory / "model.safetensors"
            first = load_file(path)
            first_hash = bridge.save_safetensors_atomic(first, path)
            second = load_file(path)
            self.assertEqual(set(second), set(first))
            for name, value in first.items():
                self.assertTrue(torch.equal(second[name], value))
            self.assertEqual(first_hash, bridge.save_safetensors_atomic(second, path))

    def test_rewrite_after_loading_releases_the_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="safetensors-rewrite-") as scratch:
            directory = Path(scratch)
            bridge.write_package(directory)
            path = directory / "model.safetensors"
            loaded = load_file(path)
            name = next(iter(loaded))
            expected = loaded[name].clone() + 1.0
            del loaded
            gc.collect()

            def mutate(tensors: dict[str, torch.Tensor]) -> None:
                tensors[name] = expected

            digest = bridge.rewrite_safetensors_atomic(path, mutate)
            self.assertEqual(len(digest), 64)
            rewritten = load_file(path)
            self.assertTrue(torch.equal(rewritten[name], expected))


if __name__ == "__main__":
    unittest.main()

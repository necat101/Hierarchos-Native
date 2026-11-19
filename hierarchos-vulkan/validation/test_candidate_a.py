#!/usr/bin/env python3
"""Negative controls for the Candidate A current-PR gate.

Every mutation starts from the genuinely passing tree and must be rejected.
All fixtures are temporary copies: no canonical shader, source or receipt file is
written by this suite.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import check_candidate_a as c

FILES = (
    tuple(spec["source"] for spec in c.CANDIDATE.values())
    + tuple(spec["binary"] for spec in c.CANDIDATE.values())
    + (c.WMMA_RS, c.WMMA_BENCH, c.ABBA_RUNNER, c.ABBA_ANALYSER)
)


class CandidateAControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="candidate-a-")
        cls.tree = Path(cls.tmp.name) / "tree"
        for relative in FILES:
            target = cls.tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.ROOT / relative, target)
        cls.receipt = json.loads(c.FIXTURE.read_text(encoding="utf-8"))
        cls.changed = set(subprocess.check_output(
            ["git", "-C", str(c.ROOT), "diff", "--name-only",
             f"{c.BASE_REVISION}...HEAD"], text=True).split())
        cls.diff = {
            path: subprocess.check_output(
                ["git", "-C", str(c.ROOT), "diff", f"{c.BASE_REVISION}...HEAD", "--", path],
                text=True)
            for path in (c.WMMA_RS, c.WMMA_BENCH)
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def mutated(self, mutate) -> Path:
        scratch = tempfile.TemporaryDirectory(prefix="candidate-a-control-")
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name) / "tree"
        shutil.copytree(self.tree, root)
        mutate(root)
        return root

    def assertRejected(self, root, *, changed=None, diff=None, reproduction=None,
                       message="the gate accepted a forged candidate"):
        with self.assertRaises(ValueError, msg=message):
            c.validate(
                root,
                changed=self.changed if changed is None else changed,
                diff=self.diff if diff is None else diff,
                reproduction=self.receipt if reproduction is None else reproduction,
            )

    def forged_receipt(self) -> dict:
        return json.loads(json.dumps(self.receipt))

    # -- positive --------------------------------------------------------
    def test_positive_candidate_validates(self):
        summary = c.validate(self.tree, changed=self.changed, diff=self.diff,
                             reproduction=self.receipt)
        self.assertEqual(sorted(summary), sorted(c.CANDIDATE))
        for module in summary.values():
            self.assertEqual(module["cooperative_mul_adds"], 4)

    # -- the qualified bytes cannot change -------------------------------
    def test_changed_shader_source_is_rejected(self):
        def mutate(root):
            path = root / c.CANDIDATE["wmma_gemm_f16_f32acc"]["source"]
            path.write_bytes(path.read_bytes() + b"// tampered\n")
        self.assertRejected(self.mutated(mutate))

    def test_changed_shader_binary_is_rejected(self):
        def mutate(root):
            path = root / c.CANDIDATE["wmma_gemm_f16_f32acc"]["binary"]
            path.write_bytes(path.read_bytes() + b"\x00\x00\x00\x00")
        self.assertRejected(self.mutated(mutate))

    def test_missing_shader_binary_is_rejected(self):
        def mutate(root):
            (root / c.CANDIDATE["wmma_gemm_f16_f16acc"]["binary"]).unlink()
        self.assertRejected(self.mutated(mutate))

    def test_packed_operand_staging_cannot_be_dropped(self):
        def mutate(root):
            path = root / c.CANDIDATE["wmma_gemm_f16_f16acc"]["source"]
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("f16vec2 pair = a[off >> 1u];", "")
                            .replace("f16vec2 pair = b[off >> 1u];", ""),
                            encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    # -- the policy surface cannot change --------------------------------
    def test_profitability_floor_cannot_move(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("16_000_000", "8_000_000"), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_boundary_shape_cannot_be_dropped(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("(19, 48, 64)", "(20, 48, 64)"), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_dispatch_or_tolerance_edit_is_rejected(self):
        forged = dict(self.diff)
        forged[c.WMMA_RS] = self.diff[c.WMMA_RS] + (
            "@@ -1,1 +1,1 @@\n"
            "-    const LIMIT: f32 = 2e-7;\n"
            "+    const LIMIT: f32 = 2e-3;\n")
        self.assertRejected(self.tree, diff=forged)

    def test_removed_comment_or_extra_hunk_is_rejected(self):
        forged = dict(self.diff)
        forged[c.WMMA_RS] = self.diff[c.WMMA_RS].replace(
            "+        // Shapes are chosen", "-        // Shapes are chosen")
        self.assertRejected(self.tree, diff=forged)

    # -- the benchmark methodology cannot change -------------------------
    def test_benchmark_statistic_edit_is_rejected(self):
        forged = dict(self.diff)
        forged[c.WMMA_BENCH] = self.diff[c.WMMA_BENCH] + (
            "@@ -1,1 +1,1 @@\n"
            "-                let round_median = median(&samples);\n"
            "+                let round_median = samples[0];\n")
        self.assertRejected(self.tree, diff=forged)

    def test_sample_export_cannot_be_dropped_from_the_benchmark(self):
        def mutate(root):
            path = root / c.WMMA_BENCH
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("--dump-samples", "--dump"), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_abba_runner_defaults_cannot_drift(self):
        def mutate(root):
            path = root / c.ABBA_RUNNER
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("PAIRS:-8", "PAIRS:-4"), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    # -- what counts as part of the change -------------------------------
    def test_protected_source_outside_candidate_is_rejected(self):
        self.assertRejected(self.tree,
                            changed=self.changed | {"hierarchos-vulkan/src/transformer.rs"})

    def test_model_weight_in_the_change_is_rejected(self):
        self.assertRejected(self.tree,
                            changed=self.changed | {"hierarchos-vulkan/models/model.safetensors"})

    def test_dependency_manifest_change_is_rejected(self):
        self.assertRejected(self.tree, changed=self.changed | {"Cargo.lock"})

    # -- the recorded reproduction is a claim, not a pass ----------------
    def test_reproduction_must_be_byte_identical(self):
        forged = self.forged_receipt()
        forged["shaders"]["wmma_gemm_f16_f32acc"]["rebuilt_sha256"] = "0" * 64
        self.assertRejected(self.tree, reproduction=forged)

    def test_reproduction_must_hold_four_multiply_adds(self):
        forged = self.forged_receipt()
        forged["shaders"]["wmma_gemm_f16_f16acc"]["cooperative_mul_adds"] = 3
        self.assertRejected(self.tree, reproduction=forged)

    def test_reproduction_flags_are_pinned(self):
        forged = self.forged_receipt()
        forged["flags"] = [flag for flag in forged["flags"] if flag != "-O"]
        self.assertRejected(self.tree, reproduction=forged)

    def test_reproduction_must_name_the_qualified_source(self):
        forged = self.forged_receipt()
        forged["shaders"]["wmma_gemm_f16_f32acc"]["source_sha256"] = "0" * 64
        self.assertRejected(self.tree, reproduction=forged)

    # -- SPIR-V structure ------------------------------------------------
    def test_module_without_four_multiply_adds_is_rejected(self):
        header = struct.pack("<5I", c.SPIRV_MAGIC, c.SPIRV_VERSION, 0, 10, 0)
        body = b"".join(
            struct.pack("<I", (1 << 16) | c.OP_COOPERATIVE_MATRIX_MUL_ADD)
            for _ in range(3))
        module = c.spirv_summary(header + body)
        self.assertEqual(module["cooperative_mul_adds"], 3)
        with self.assertRaises(ValueError):
            c.check_spirv_structure(module, "forged")

    def test_non_spirv_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            c.spirv_summary(b"not a module at all")

    def test_instruction_stream_must_tile_the_module(self):
        header = struct.pack("<5I", c.SPIRV_MAGIC, c.SPIRV_VERSION, 0, 10, 0)
        body = struct.pack("<I", (4 << 16) | c.OP_COOPERATIVE_MATRIX_MUL_ADD)
        with self.assertRaises(ValueError):
            c.spirv_summary(header + body)


if __name__ == "__main__":
    unittest.main()

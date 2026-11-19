#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXIV current-candidate gate.

Every mutation starts from the genuinely passing tree and receipt, and must be
rejected. Fixtures are temporary copies: no canonical shader, source or receipt
file is written by this suite.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import check_current_candidate as c

FILES = (
    tuple(spec["source"] for spec in c.CURRENT.values())
    + tuple(spec["binary"] for spec in c.CURRENT.values())
    + (c.WMMA_RS, c.WMMA_QUALIFY)
)


class CurrentCandidateControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase-lxiv-")
        cls.tree = Path(cls.tmp.name) / "tree"
        for relative in FILES:
            target = cls.tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.ROOT / relative, target)
        cls.receipt = json.loads(c.FIXTURE.read_text(encoding="utf-8"))
        cls.changed = set(subprocess.check_output(
            ["git", "-C", str(c.ROOT), "diff", "--name-only",
             f"{c.BASE_REVISION}...HEAD"], text=True).split())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def mutated(self, mutate) -> Path:
        scratch = tempfile.TemporaryDirectory(prefix="phase-lxiv-control-")
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name) / "tree"
        shutil.copytree(self.tree, root)
        mutate(root)
        return root

    def forged(self) -> dict:
        return json.loads(json.dumps(self.receipt))

    def assertRejected(self, root=None, *, changed=None, receipt=None,
                       message="the gate accepted a forged implementation"):
        with self.assertRaises(ValueError, msg=message):
            c.validate(
                self.tree if root is None else root,
                changed=self.changed if changed is None else changed,
                receipt=self.receipt if receipt is None else receipt,
            )

    # -- positive --------------------------------------------------------
    def test_positive_current_candidate_validates(self):
        summary = c.validate(self.tree, changed=self.changed, receipt=self.receipt)
        self.assertEqual(summary["wmma_gemm_f16_f16acc"]["cooperative_mul_adds"], 8)
        self.assertEqual(summary["wmma_gemm_f16_f32acc"]["cooperative_mul_adds"], 4)

    # -- the qualified bytes cannot change -------------------------------
    def test_changed_performance_shader_is_rejected(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f32acc"]["binary"]
            path.write_bytes(path.read_bytes() + b"\x00\x00\x00\x00")
        self.assertRejected(self.mutated(mutate))

    def test_changed_split_source_is_rejected(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            path.write_bytes(path.read_bytes() + b"// tampered\n")
        self.assertRejected(self.mutated(mutate))

    def test_missing_split_binary_is_rejected(self):
        def mutate(root):
            (root / c.CURRENT["wmma_gemm_f16_f16acc"]["binary"]).unlink()
        self.assertRejected(self.mutated(mutate))

    # -- the accuracy mechanism cannot be removed ------------------------
    def test_second_accumulator_bank_cannot_be_dropped(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            path.write_text(path.read_text(encoding="utf-8").replace("alt00", "b00"),
                            encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_alternating_slice_selector_cannot_be_dropped(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            path.write_text(
                path.read_text(encoding="utf-8")
                    .replace("if ((slice & 1u) == 0u)", "if (true)"),
                encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_fp16_merge_cannot_become_a_widening_merge(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            path.write_text(
                path.read_text(encoding="utf-8")
                    .replace("float16_t merged = float16_t(c_part[e] + float(c_tile[e]));",
                             "float merged = c_part[e] + float(c_tile[e]);"),
                encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_fp32_accumulator_in_the_fp16_arm_is_rejected(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            text = path.read_text(encoding="utf-8")
            text += ("\ncoopmat<float, gl_ScopeSubgroup, 16, 16, "
                     "gl_MatrixUseAccumulator> widened;\n")
            path.write_text(text, encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_packed_operand_staging_cannot_be_dropped(self):
        def mutate(root):
            path = root / c.CURRENT["wmma_gemm_f16_f16acc"]["source"]
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("f16vec2 pair = a[off >> 1u];", "")
                            .replace("f16vec2 pair = b[off >> 1u];", ""),
                            encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    # -- the policy surface cannot change --------------------------------
    def test_numerical_limit_cannot_be_relaxed(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("FULL_VS_FP32_MAX_ABS: f64 = 5.0e-2",
                                     "FULL_VS_FP32_MAX_ABS: f64 = 5.0e-1"),
                            encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_profitability_floor_cannot_move(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("16_000_000", "8_000_000"), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_matched_oracle_variant_cannot_be_removed(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("    Fp16Split2,\n", ""), encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_oracle_option_cannot_be_removed_from_the_qualifier(self):
        def mutate(root):
            path = root / c.WMMA_QUALIFY
            path.write_text(path.read_text(encoding="utf-8")
                            .replace('Some("split2")', 'Some("other")'),
                            encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    def test_long_reduction_accuracy_test_cannot_be_removed(self):
        def mutate(root):
            path = root / c.WMMA_RS
            path.write_text(
                path.read_text(encoding="utf-8")
                    .replace("wmma_full_fp16_meets_fp32_budget_on_long_reductions",
                             "wmma_full_fp16_smoke"),
                encoding="utf-8")
        self.assertRejected(self.mutated(mutate))

    # -- the recorded numerics are a claim, not a pass -------------------
    def test_contract_matched_run_cannot_record_a_failure(self):
        forged = self.forged()
        forged["runs"]["full-fp16-split2"]["failed"] = 1
        self.assertRejected(receipt=forged)

    def test_historical_failure_count_cannot_be_zeroed(self):
        forged = self.forged()
        forged["runs"]["full-fp16-tile16"]["failed"] = 0
        self.assertRejected(receipt=forged)

    def test_a_different_four_cannot_be_reported(self):
        forged = self.forged()
        rows = forged["runs"]["full-fp16-tile16"]["results"]
        for row in rows:
            if row["label"] == "128x128x96 wmma-full-fp16 vs matched oracle":
                row["passed"] = True
            if row["label"] == "33x17x19 wmma-full-fp16 vs matched oracle":
                row["passed"] = False
        self.assertRejected(receipt=forged)

    def test_fp32_budget_gate_cannot_be_recorded_as_failing(self):
        forged = self.forged()
        rows = forged["runs"]["full-fp16-tile16"]["results"]
        for row in rows:
            if row["label"] == "128x128x96 trans_a=true trans_b=false vs FP32":
                row["passed"] = False
        self.assertRejected(receipt=forged)

    def test_measured_value_above_its_recorded_limit_is_rejected(self):
        forged = self.forged()
        rows = forged["runs"]["full-fp16-tile16"]["results"]
        for row in rows:
            if row["label"] == "128x128x96 trans_a=true trans_b=false vs FP32":
                row["max_abs"] = 0.0731048584
        self.assertRejected(receipt=forged)

    def test_improvement_cannot_be_claimed_for_a_passing_baseline(self):
        forged = self.forged()
        forged["previously_failing"]["128x128x96 trans_a=true trans_b=false vs FP32"]["before"] = 0.01
        self.assertRejected(receipt=forged)

    def test_improvement_cannot_claim_a_relaxed_limit(self):
        forged = self.forged()
        forged["previously_failing"]["128x128x96 trans_a=true trans_b=false vs FP32"]["limit"] = 1.0
        self.assertRejected(receipt=forged)

    def test_recorded_artifact_must_match_the_committed_binary(self):
        forged = self.forged()
        forged["shaders"]["wmma_gemm_f16_f16acc"]["binary_sha256"] = "0" * 64
        self.assertRejected(receipt=forged)

    def test_reproduction_must_be_byte_identical(self):
        forged = self.forged()
        forged["shaders"]["wmma_gemm_f16_f16acc"]["rebuilt_sha256"] = "1" * 64
        self.assertRejected(receipt=forged)

    def test_production_flags_are_pinned(self):
        forged = self.forged()
        forged["flags"] = [flag for flag in forged["flags"] if flag != "-O"]
        self.assertRejected(receipt=forged)

    # -- what counts as part of the change -------------------------------
    def test_protected_source_outside_the_change_is_rejected(self):
        self.assertRejected(changed=self.changed | {"hierarchos-vulkan/src/transformer.rs"})

    def test_model_weight_in_the_change_is_rejected(self):
        self.assertRejected(changed=self.changed | {"hierarchos-vulkan/models/model.safetensors"})

    def test_dependency_manifest_change_is_rejected(self):
        self.assertRejected(changed=self.changed | {"Cargo.lock"})

    # -- SPIR-V structure ------------------------------------------------
    def test_wrong_multiply_add_count_is_rejected(self):
        header = struct.pack("<5I", c.SPIRV_MAGIC, c.SPIRV_VERSION, 0, 10, 0)
        body = b"".join(
            struct.pack("<I", (1 << 16) | c.OP_COOPERATIVE_MATRIX_MUL_ADD)
            for _ in range(7))
        module = c.spirv_summary(header + body)
        self.assertEqual(module["cooperative_mul_adds"], 7)
        # The split module must hold exactly eight; seven is a different kernel.
        self.assertNotEqual(module["cooperative_mul_adds"],
                            c.CURRENT["wmma_gemm_f16_f16acc"]["cooperative_mul_adds"])

    def test_non_spirv_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            c.spirv_summary(b"not a module at all")


if __name__ == "__main__":
    unittest.main()

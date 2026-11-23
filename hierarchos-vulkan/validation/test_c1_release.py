#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXXXVIII C1 release gate.

Every mutation starts from the genuinely passing release tree and must be
rejected. The mutations are applied to a temporary copy of the files the gate
reads; no committed shader, artifact or source file is written by this suite.

The controls are the claims the promotion rests on: the C1 bytes are the
recovered bytes, only the production P3 variant can reach them, the geometry
contract cannot lose a condition, the non-proven fallback cannot disappear, no
knob can force the staging route, and the v0.2.5 gate this release inherits
cannot be bypassed.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import check_c1_release as c
import check_p3_release as p3

FILES = (
    c.WMMA_RS,
    p3.PROJECTION_RS,
    p3.TRANSFORMER_RS,
    p3.P3["source"],
    p3.P3["binary"],
    p3.MIXED["source"],
    p3.MIXED["binary"],
    c.C1["source"],
    c.C1["binary"],
    p3.FALLBACKS["split8"]["source"],
    p3.FALLBACKS["split8"]["binary"],
    p3.FALLBACKS["two-bank"]["source"],
    p3.FALLBACKS["two-bank"]["binary"],
    p3.FALLBACKS["two-wave"]["source"],
    p3.FALLBACKS["two-wave"]["binary"],
)


class C1ReleaseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase-lxxxviii-c1-")
        cls.tree = Path(cls.tmp.name) / "tree"
        for relative in FILES:
            target = cls.tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.ROOT / relative, target)
        # The unmutated copy must pass before any control is meaningful.
        c.validate(cls.tree)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def edit(self, relative, transform):
        """Apply ``transform`` to the file's text, restoring the bytes afterwards."""
        path = self.tree / relative
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(transform(original.decode("utf-8")).encode("utf-8"))

    def append_byte(self, relative):
        # Kept separate from `edit`: appending a byte is a byte-level mutation.
        path = self.tree / relative
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(original + b"\x0a")

    def assertRejected(self, message):
        with self.assertRaises(ValueError, msg=message):
            c.validate(self.tree)

    # -- positive --------------------------------------------------------
    def test_genuine_release_passes(self):
        c.validate(self.tree)

    # -- C1 artifact identity -------------------------------------------
    def test_altered_c1_source_bytes(self):
        self.append_byte(c.C1["source"])
        self.assertRejected("an altered C1 shader source was accepted")

    def test_altered_c1_spirv_bytes(self):
        self.append_byte(c.C1["binary"])
        self.assertRejected("an altered C1 SPIR-V module was accepted")

    def test_c1_module_not_embedded(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.spv")',
            'include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave_p3.spv")'))
        self.assertRejected("a C1 module embedded from another artifact was accepted")

    # -- variant containment ---------------------------------------------
    def test_p3_variant_loses_the_staging_module(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "Self::P3 => Some(WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV),",
            "Self::P3 => None,"))
        self.assertRejected("a production variant that cannot reach C1 was accepted")

    def test_non_p3_variant_gains_the_staging_module(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "Self::Split8 | Self::TwoBank | Self::TwoWave => None,\n            Self::P3 => Some(",
            "Self::Split8 | Self::TwoBank => None,\n            Self::TwoWave => Some("
            "WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV),\n            Self::P3 => Some(", 1))
        self.assertRejected("a non-P3 variant able to dispatch C1 was accepted")

    def test_staging_shader_name_reassigned(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'Self::P3 => Some("wmma_gemm_f16_f16acc_twowave_p3_c1"),',
            'Self::P3 => Some("wmma_gemm_f16_f16acc_twowave_p3"),'))
        self.assertRejected("a staging module reporting another shader name was accepted")

    # -- predicate conditions --------------------------------------------
    def test_tile_coverage_condition_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "if geometry.rows % 32 != 0 || geometry.columns % 32 != 0 || "
            "geometry.reduction % 16 != 0 {\n        return false;\n    }",
            "if false {\n        return false;\n    }"))
        self.assertRejected("a predicate that stops requiring full tiles was accepted")

    def test_packed_pair_parity_condition_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "if lda % 2 != 0 || ldb % 2 != 0 {\n        return false;\n    }",
            "if false {\n        return false;\n    }"))
        self.assertRejected("a predicate that stops requiring even strides was accepted")

    def test_capacity_bounds_condition_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "a_bytes <= geometry.a_capacity_bytes as u64 && "
            "b_bytes <= geometry.b_capacity_bytes as u64",
            "true"))
        self.assertRejected("a predicate that stops bounding the operands was accepted")

    def test_32_bit_ceiling_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "if a_last > ceiling || b_last > ceiling {\n        return false;\n    }",
            "if false {\n        return false;\n    }"))
        self.assertRejected("a predicate that stops bounding the 32-bit index was accepted")

    # -- production wiring ----------------------------------------------
    def test_masked_fallback_removed(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "                    _ => masked,\n", "                    _ => batched,\n"))
        self.assertRejected("a selector with no P3 fallback was accepted")

    def test_staging_counter_removed(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "                    self.staging_masked.fetch_add(1, Ordering::Relaxed);",
            "                    self.plain_dispatches.fetch_add(1, Ordering::Relaxed);"))
        self.assertRejected("a selector that no longer counts its fallback was accepted")

    def test_staging_geometry_capacity_unbound(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "a_capacity_bytes: a.capacity_bytes(),",
            "a_capacity_bytes: usize::MAX,"))
        self.assertRejected("a selector not bound to the real operand capacity was accepted")

    # -- no forced route -------------------------------------------------
    def test_forced_staging_environment_variable(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'pub const FULL_FP16_VARIANT_ENV: &str = "HIERARCHOS_WMMA_FULL_FP16_VARIANT";',
            'pub const FULL_FP16_VARIANT_ENV: &str = "HIERARCHOS_WMMA_FULL_FP16_VARIANT";\n'
            'pub const HIERARCHOS_WMMA_C1: &str = "HIERARCHOS_WMMA_C1";'))
        self.assertRejected("an environment variable that can force C1 was accepted")

    def test_variant_parser_exposes_c1(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            '"p3" | "twowave-p3" | "two-wave-p3" => Ok(Self::P3),',
            '"p3" | "twowave-p3" | "two-wave-p3" | "c1" => Ok(Self::P3),'))
        self.assertRejected("a staging implementation exposed as a variant name was accepted")

    # -- inherited v0.2.5 gate -------------------------------------------
    def test_v025_gate_cannot_be_bypassed(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV,",
            "Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_SPV,"))
        self.assertRejected("a tree the v0.2.5 gate rejects was accepted by v0.2.6")

    def test_altered_p3_source_bytes(self):
        self.append_byte(p3.P3["source"])
        self.assertRejected("an altered P3 shader source was accepted by v0.2.6")

    # -- source structure -------------------------------------------------
    def test_extra_barrier_in_c1_source(self):
        self.edit(c.C1["source"], lambda text: text.replace(
            "                    barrier();\n\n                    coopMatLoad",
            "                    barrier();\n                    barrier();\n\n"
            "                    coopMatLoad", 1))
        self.assertRejected("a C1 source with a changed barrier count was accepted")

    def test_fast_path_leaked_into_p3(self):
        self.edit(c.C1["source"], lambda text: text.replace(
            "if (wide_a && m0 + 32u <= pc.m && k0 + 16u <= pc.k) {",
            "if (wide_a && m0 + 32u <= pc.n && k0 + 16u <= pc.k) {"))
        self.assertRejected("a fast-path predicate that changed its bound was accepted")


if __name__ == "__main__":
    unittest.main()

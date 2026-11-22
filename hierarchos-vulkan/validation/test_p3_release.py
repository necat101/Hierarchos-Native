#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXXXI P3 release gate.

Every mutation starts from the genuinely passing release tree and must be
rejected. The mutations are applied to a temporary copy of the files the gate
reads; no committed shader, artifact or source file is written by this suite.

The three controls the release contract calls out explicitly are
`test_altered_p3_source_bytes`, `test_altered_p3_spirv_bytes`,
`test_removed_production_admission` and
`test_unintended_qualification_route_exposure`. The rest cover the remaining
claims the gate makes, so a weakened predicate cannot pass as the release.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import check_p3_release as c

SPLIT8 = c.FALLBACKS["split8"]
TWO_BANK = c.FALLBACKS["two-bank"]
TWO_WAVE = c.FALLBACKS["two-wave"]

FILES = (
    c.WMMA_RS,
    c.PROJECTION_RS,
    c.TRANSFORMER_RS,
    c.P3["source"],
    c.P3["binary"],
    c.MIXED["source"],
    c.MIXED["binary"],
    SPLIT8["source"],
    SPLIT8["binary"],
    TWO_BANK["source"],
    TWO_BANK["binary"],
    TWO_WAVE["source"],
    TWO_WAVE["binary"],
)


class P3ReleaseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase-lxxxi-p3-")
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
        """Apply ``transform`` to the file's text, restoring the bytes afterwards.

        The bytes are read and written directly so the temporary copy keeps the
        committed line endings; the gate reads through universal newlines, so a
        mutation that only re-encodes the file cannot masquerade as a failure.
        """
        path = self.tree / relative
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(transform(original.decode("utf-8")).encode("utf-8"))

    def append_byte(self, relative):
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

    # -- P3 artifact identity --------------------------------------------
    def test_altered_p3_source_bytes(self):
        self.append_byte(c.P3["source"])
        self.assertRejected("an altered P3 shader source was accepted")

    def test_altered_p3_spirv_bytes(self):
        self.append_byte(c.P3["binary"])
        self.assertRejected("an altered P3 SPIR-V module was accepted")

    def test_p3_module_no_longer_embedded(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "include_bytes!(\"../shaders/wmma_gemm_f16_f16acc_twowave_p3.spv\")",
            "include_bytes!(\"../shaders/wmma_gemm_f16_f16acc_twowave.spv\")"))
        self.assertRejected("a P3 variant embedding a different module was accepted")

    def test_p3_variant_dispatching_another_module(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV,",
            "Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_SPV,"))
        self.assertRejected("a P3 variant dispatching another module was accepted")

    def test_p3_shader_name_reassigned(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'Self::P3 => "wmma_gemm_f16_f16acc_twowave_p3",',
            'Self::P3 => "wmma_gemm_f16_f16acc_twowave",'))
        self.assertRejected("a P3 variant reporting another shader name was accepted")

    # -- variant exposure ------------------------------------------------
    def test_unintended_qualification_route_exposure(self):
        """A fifth, qualification-only variant must not become selectable."""
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "#[default]\n    P3,\n}",
            "#[default]\n    P3,\n    Qualification,\n}", 1))
        self.assertRejected("an extra qualification-only variant was accepted")

    def test_qualification_name_accepted_by_the_selector(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'assert!(FullFp16Variant::from_env_value(Some("experimental".to_owned())).is_err());',
            'assert!(FullFp16Variant::from_env_value(Some("experimental".to_owned())).is_ok());'))
        self.assertRejected("the selector was allowed to name a qualification-only module")

    def test_p3_is_no_longer_the_default(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "#[default]\n    P3,", "    P3,", 1))
        self.assertRejected("a Full FP16 default other than P3 was accepted")

    def test_committed_variant_inventory_shrunk(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "pub const ALL: [Self; 4] = [Self::Split8, Self::TwoBank, Self::TwoWave, Self::P3];",
            "pub const ALL: [Self; 3] = [Self::Split8, Self::TwoWave, Self::P3];"))
        self.assertRejected("a dropped committed variant was accepted")

    # -- opt-in mode -----------------------------------------------------
    def test_wmma_stops_being_opt_in(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "_ => Ok(Self::Off),", "_ => Ok(Self::Mixed),", 1))
        self.assertRejected("an opt-out default WMMA mode was accepted")

    # -- profitability policy --------------------------------------------
    def test_threshold_moved(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "pub const WMMA_MIN_PROFITABLE_MACS: u64 = 16_000_000;",
            "pub const WMMA_MIN_PROFITABLE_MACS: u64 = 8_000_000;"))
        self.assertRejected("a moved profitability floor was accepted")

    def test_profitability_predicate_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "            >= WMMA_MIN_PROFITABLE_MACS",
            "            >= 1"))
        self.assertRejected("a profitability predicate that admits every shape was accepted")

    def test_shape_eligibility_dropped(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "    wmma_shape_eligible(rows, columns, reduction)\n        && (rows as u64)",
            "    (rows as u64)"))
        self.assertRejected("a predicate without shape eligibility was accepted")

    def test_ineligible_shape_dispatches_cooperatively(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            "                    Some(WmmaGemmArm::PlainF16Storage)",
            "                    Some(WmmaGemmArm::FullFp16Accumulate)", 1))
        self.assertRejected("an ineligible shape reaching a cooperative arm was accepted")

    # -- production admission -------------------------------------------
    def test_removed_production_admission(self):
        """The projection constructor must keep applying the predicate."""
        self.edit(c.PROJECTION_RS, lambda text: text.replace(
            "wmma::wmma_shape_profitable(max_rows, output_dim, input_dim)",
            "wmma::wmma_shape_eligible(max_rows, output_dim, input_dim)"))
        self.assertRejected("a projection gate without the profitability predicate was accepted")

    def test_removed_transformer_admission(self):
        self.edit(c.TRANSFORMER_RS, lambda text: text.replace(
            "if !crate::wmma::wmma_shape_profitable(rows, output_dim, input_dim) {",
            "if !crate::wmma::wmma_shape_eligible(rows, output_dim, input_dim) {"))
        self.assertRejected("a Transformer gate without the profitability predicate was accepted")

    def test_dropped_recorded_batch_admission(self):
        self.edit(c.PROJECTION_RS, lambda text: text.replace(
            ".filter(|_| ProjectionWmma::admits(rows, self.input_dim, self.output_dim))",
            ".filter(|_| true)", 1))
        self.assertRejected("a recorded-batch path that lost its admission gate was accepted")

    def test_declined_projection_stops_recording_its_fallback(self):
        self.edit(c.PROJECTION_RS, lambda text: text.replace(
            "                crate::wmma_linear::note_policy_fallback();\n", "", 1))
        self.assertRejected("a declined projection without its fallback record was accepted")

    # -- Mixed and fallbacks ---------------------------------------------
    def test_altered_mixed_artifact(self):
        self.append_byte(c.MIXED["binary"])
        self.assertRejected("an altered Mixed artifact was accepted")

    def test_removed_two_wave_fallback(self):
        (self.tree / TWO_WAVE["binary"]).unlink()
        self.addCleanup(shutil.copy2, c.ROOT / TWO_WAVE["binary"],
                        self.tree / TWO_WAVE["binary"])
        self.assertRejected("a removed two-wave fallback was accepted")

    def test_two_bank_fallback_bytes_changed(self):
        self.append_byte(TWO_BANK["binary"])
        self.assertRejected("altered two-bank fallback bytes were accepted")

    def test_fallback_selection_variable_removed(self):
        self.edit(c.WMMA_RS, lambda text: text.replace(
            'pub const FULL_FP16_VARIANT_ENV: &str = "HIERARCHOS_WMMA_FULL_FP16_VARIANT";',
            'pub const FULL_FP16_VARIANT_ENV: &str = "";'))
        self.assertRejected("a removed fallback selection variable was accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)

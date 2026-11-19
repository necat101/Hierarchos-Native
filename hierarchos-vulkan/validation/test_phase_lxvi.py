#!/usr/bin/env python3
"""Positive and negative controls for the Phase LXVI gate.

Each control starts from the genuinely passing evidence, mutates one thing, and
requires the gate to reject it. Mutated copies live in a temporary directory;
no receipt, capture, sample or shard in the tree is written by this suite.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import abba_stats
import check_phase_lxvi as c


class PhaseLxviControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="phase-lxvi-")
        self.root = Path(self.tmp.name)
        self.saved = {name: getattr(c, name) for name in
                      ("CORRECTION", "SIMULATION", "RGP_RECORD", "TRAINING_RECORD")}

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(c, name, value)
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def copy_json(self, source: Path, name: str) -> Path:
        target = self.root / name
        shutil.copy2(source, target)
        return target

    def edit(self, path: Path, mutate):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def assertRejected(self):
        with self.assertRaises(ValueError):
            c.validate()

    # -- positive control -------------------------------------------------
    def test_committed_evidence_passes(self):
        result = c.validate()
        self.assertGreater(result["correction_entries"], 0)

    # -- statistical controls ---------------------------------------------
    def test_forged_quantile_is_rejected(self):
        original = abba_stats.t_critical
        abba_stats.t_critical = lambda df, confidence=0.95: 1.96
        try:
            self.assertRejected()
        finally:
            abba_stats.t_critical = original

    def test_unsupported_sample_count_fails_closed(self):
        with self.assertRaises(abba_stats.StatisticsError):
            abba_stats.t_critical(abba_stats.MAX_DF + 1)

    def test_forged_confidence_interval_is_rejected(self):
        path = self.copy_json(self.saved["CORRECTION"], "correction.json")
        self.edit(path, lambda r: r["entries"][0].__setitem__("corrected_ci95", [0.5, 1.5]))
        c.CORRECTION = path
        self.assertRejected()

    def test_narrower_than_normal_interval_is_rejected(self):
        path = self.copy_json(self.saved["CORRECTION"], "correction2.json")

        def shrink(record):
            entry = record["entries"][0]
            lo, hi = entry["corrected_ci95"]
            mid = (lo + hi) / 2
            entry["corrected_ci95"] = [mid - 1e-6, mid + 1e-6]

        self.edit(path, shrink)
        c.CORRECTION = path
        self.assertRejected()

    def test_missing_unrecomputable_sweeps_is_rejected(self):
        path = self.copy_json(self.saved["CORRECTION"], "correction3.json")
        self.edit(path, lambda r: r.__setitem__("unrecomputable_sweeps", []))
        c.CORRECTION = path
        self.assertRejected()

    # -- K=96 simulation controls -----------------------------------------
    def test_missing_seed_outputs_are_rejected(self):
        path = self.copy_json(self.saved["SIMULATION"], "sim.json")
        self.edit(path, lambda r: r.__setitem__(
            "results",
            [row for row in r["results"] if row["seed"] != r["seeds"][0]],
        ))
        c.SIMULATION = path
        self.assertRejected()

    def test_dropped_seed_is_rejected(self):
        path = self.copy_json(self.saved["SIMULATION"], "sim2.json")
        self.edit(path, lambda r: r["seeds"].pop())
        c.SIMULATION = path
        self.assertRejected()

    def test_altered_tolerance_is_rejected(self):
        path = self.copy_json(self.saved["SIMULATION"], "sim3.json")

        def loosen(record):
            for row in record["results"]:
                row["oracle_limit"] = 0.05

        self.edit(path, loosen)
        c.SIMULATION = path
        self.assertRejected()

    def test_margin_target_claim_is_rejected(self):
        path = self.copy_json(self.saved["SIMULATION"], "sim4.json")

        def claim(record):
            target = record["target_for_25pct_headroom"]
            for row in record["results"]:
                if row["model"] == "split8-roundrobin-pairwise":
                    row["max_abs_vs_tile16"] = 0.05
                    row["target_met_vs_tile16"] = True

        self.edit(path, claim)
        c.SIMULATION = path
        self.assertRejected()

    def test_simulation_relabelled_as_hardware_is_rejected(self):
        path = self.copy_json(self.saved["SIMULATION"], "sim5.json")
        self.edit(path, lambda r: r.__setitem__("kind", "hardware measurement"))
        c.SIMULATION = path
        self.assertRejected()

    # -- hardware-evidence controls ---------------------------------------
    def test_forged_hit_counts_without_capture_hash_are_rejected(self):
        path = self.root / "rgp.json"
        path.write_text(json.dumps({
            "rgp_version": "2.7.1.13",
            "adapter": "Radeon 780M",
            "event_id": 1,
            "pipeline_hash": "deadbeef",
            "static_wmma": 32,
            "sampling_scope": "single_shader_engine",
            "total_wavefronts": 64,
            "analyzed_wavefronts": 64,
            "per_instruction": [{"instruction": "v_wmma_f16_16x16x16_f16", "hit_count": 4}],
        }), encoding="utf-8")
        c.RGP_RECORD = path
        self.assertRejected()

    def test_whole_gpu_extrapolation_is_rejected(self):
        path = self.root / "rgp2.json"
        path.write_text(json.dumps({
            "rgp_version": "2.7.1.13",
            "adapter": "Radeon 780M",
            "capture_sha256": "0" * 64,
            "capture_bytes": 1,
            "event_id": 1,
            "pipeline_hash": "deadbeef",
            "static_wmma": 32,
            "sampling_scope": "single_shader_engine",
            "total_wavefronts": 64,
            "analyzed_wavefronts": 64,
            "per_instruction": [{"instruction": "v_wmma_f16_16x16x16_f16", "hit_count": 4}],
            "whole_gpu_normalisation": "hit_count * 64",
        }), encoding="utf-8")
        c.RGP_RECORD = path
        self.assertRejected()

    def test_missing_evidence_is_reported_not_claimed(self):
        c.RGP_RECORD = self.root / "absent-rgp.json"
        c.TRAINING_RECORD = self.root / "absent-training.json"
        result = c.validate()
        joined = " ".join(result["missing_evidence"])
        self.assertIn("RGP", joined)
        self.assertIn("training-step", joined)


if __name__ == "__main__":
    unittest.main()

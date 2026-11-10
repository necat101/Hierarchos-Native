#!/usr/bin/env python3
"""Negative controls for the Phase XVI current-source qualification.

Every control starts from a receipt that genuinely passes, so a rejection below
means the guard noticed the mutation rather than the fixture being invalid.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_phase16_receipts import (
    CURRENT_RECEIPT, ENVIRONMENT, EXTRA_SOURCES, LIMIT, REQUIRED_EVIDENCE, SCOPE,
    SEALED_GATE, SHADERS, digest, verify_receipt,
)

GEOMETRY = {
    "fixture-causal": (8, 8, 64, 128, 1, False),
    "fixture-full": (8, 8, 64, 128, 0, False),
    "padded-causal": (2, 2, 16, 64, 1, True),
    "small-causal": (1, 1, 4, 4, 1, False),
}
HEAD = "1" * 40


def test_log() -> str:
    lines = []
    for case, (batch, heads, seq, head_dim, causal, masked) in GEOMETRY.items():
        for tensor in "qkv":
            elements = batch * heads * seq * head_dim
            lines.append(
                f"[{case}] batch={batch} heads={heads} seq={seq} head_dim={head_dim} "
                f"causal={causal} masked={'true' if masked else 'false'} unwritten=0 "
                f"grad_{tensor}: elements={elements} bitwise_equal={elements} "
                f"max_abs=0.000e0 max_rel=0.000e0 rms=0.000e0 max_ulp=0 "
                f"nan(expected/actual)=0/0 inf=0/0 first_divergence=none PASS"
            )
    return "\n".join(lines) + "\n"


def harness_log() -> str:
    return "\n".join([
        "=== negative controls ===",
        "negative controls: 11/11 detected",
        "=== training arms ===",
        "scalar 2 0",
        "twopass 2 0",
        "=== dispatch counts (profile lines) ===",
        "scalar arm : scalar=2 query=0 key=0",
        "twopass arm: scalar=0 query=2 key=2",
        "two-pass backward: kernels=true scratch=true supported=true key_blocks=1 "
        "causal=1 unfused_qk=9 head_dim=128 kv_seq=64",
        f"scalar  adapter_config.json {'a' * 64}",
        f"scalar  adapter_model.safetensors {'b' * 64}",
        f"twopass adapter_config.json {'a' * 64}",
        f"twopass adapter_model.safetensors {'b' * 64}",
        "=== PASS: both passes dispatched and every adapter file is bit-identical",
    ]) + "\n"


def shader_sections():
    sealed = json.loads(SEALED_GATE.read_text())
    historical = [dict(artifact=row["artifact"], sha256=row["sha256"],
                       target_env=row["target_env"]) for row in sealed["artifacts"]]
    rebuilt = []
    for name in ("transformer_attention_backward_query_rdna3.spv",
                 "transformer_attention_backward_key_rdna3.spv"):
        sha = digest(SHADERS / name)
        rebuilt.append(dict(artifact=name, sha256=sha, rebuilt_sha256=sha,
                            target_env="vulkan1.2", compiler="glslc",
                            compiler_version="shaderc v2023.8", validated=True))
    return dict(
        inventory=45, status_counts={"identical": 45},
        historical=dict(artifacts=43, verified=43, recompiled=0,
                        sealed_gate_sha256=digest(SEALED_GATE), rows=historical),
        phase16=dict(artifacts=2, rebuilt=2, validated=2, rows=rebuilt),
    )


FAMILY_COUNTS = {"shapes": 16, "shared": 74, "kimi": 3, "seq2seq": 6,
                 "gemma3": 4, "llava": 4, "gemma4-legacy": 1, "gemma4-rgb": 1}


def family(count, prefix="case"):
    return dict(target="rdna3", complete=True, **{"pass": True}, rtol=0,
                comparison="strict_less_than", absolute_limit=LIMIT,
                expected_cases=count, completed_cases=count,
                results=[dict(case=f"{prefix}-{index}", **{"pass": True},
                              checks=[dict(metric="final/logits", max_abs=5.96e-08,
                                           **{"pass": True})])
                         for index in range(count)])


def _evidence_payload(name: str) -> str:
    if name == "two-pass/test.log":
        return test_log()
    if name == "two-pass/harness.log":
        return harness_log()
    if name == "primitives.log":
        body = "\n".join(f"test rdna3_gpt2_{index} ... ok" for index in range(9))
        return body + "\ntest result: ok. 9 passed; 0 failed; 1 ignored\n"
    if name == "isolation.json":
        return json.dumps(dict(unchanged=True, files_checked=1341,
                               existing_recipes=672, failures=[]))
    if name == "v41native/qualification.json":
        return json.dumps(dict(**{"pass": True}, expected_tests=30, completed_tests=30,
                               jobs=[dict(**{"pass": True})], max_abs_logit=0.0,
                               binary_sha256="a" * 64))
    if name == "v41bridge/qualification.json":
        return json.dumps(dict(**{"pass": True}, expected_cases=2, completed_cases=2,
                               binary_sha256="a" * 64))
    if name.endswith("qualification.json"):
        domain = name.split("/")[0]
        return json.dumps(family(FAMILY_COUNTS[domain], domain))
    return "{}\n"


def build(directory: Path):
    """Write a genuinely passing receipt plus the evidence it ships."""
    from emit_two_pass_evidence import parse

    evidence_dir = directory / "evidence"
    for name in REQUIRED_EVIDENCE:
        path = evidence_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_evidence_payload(name))
    evidence = {name: digest(evidence_dir / name) for name in REQUIRED_EVIDENCE}
    gradient = parse(evidence_dir / "two-pass/test.log", evidence_dir / "two-pass/harness.log")
    backward = dict(
        enabled_by=ENVIRONMENT, scalar_default=True,
        cases=len({row["case"] for row in gradient["cases"]}),
        comparisons=gradient["comparisons"], elements=gradient["elements"],
        bitwise_equal=gradient["bitwise_equal"], max_abs=gradient["max_abs"],
        max_ulp=gradient["max_ulp"], unwritten=gradient["unwritten"],
        adapters_identical=True, adapter_files=["adapter_config.json " + "a" * 64],
        dispatches=gradient["dispatches"], diagnostic=gradient["diagnostic"],
        negative_controls=gradient["negative_controls"],
        test_log_sha256=gradient["evidence"]["test_log"],
        harness_log_sha256=gradient["evidence"]["harness_log"],
    )
    sources = {f"hierarchos-vulkan/src/fake{i}.rs": "a" * 64 for i in range(134)}
    sources.update({name: "a" * 64 for name in EXTRA_SOURCES})
    receipt = dict(
        schema=2, **{"pass": True}, target="rdna3", scope=SCOPE, source_head=HEAD,
        worktree=True, absolute_limit=LIMIT, rtol=0, comparison="strict_less_than",
        adapter=dict(vendor="1002", device="15bf", driver="32.0.31007.6002"),
        binaries={name: "a" * 64 for name in ("image_text_pipeline_probe",
                                              "gemma3_vision_probe", "gemma4_vision_probe",
                                              "llava_pipeline_probe", "libtest")},
        producer_sha256="a" * 64, cpu_fixtures_sha256="a" * 64,
        shapes=dict(cases=16, worst_logit=5.96e-08, binary_sha256="a" * 64,
                    fixtures_sha256="a" * 64),
        retained={name: dict(cases=count, pass_=True, qualification_sha256="a" * 64,
                             worst_logit=1e-7)
                  for name, count in (("shared", 74), ("kimi", 3), ("seq2seq", 6),
                                      ("gemma3", 4), ("gemma4", 2), ("llava", 4))},
        primitives=dict(passed=9, failed=0, ignored=1, log_sha256="a" * 64),
        v41=dict(native=dict(tests=30, max_abs_logit=0.0, receipt_sha256="a" * 64),
                 bridge=dict(cases=2, receipt_sha256="a" * 64)),
        shaders=shader_sections(), attention_backward=backward,
        isolation=dict(files_checked=1341, existing_recipes=672, receipt_sha256="a" * 64),
        sources=sources, evidence=evidence,
        reused=[dict(evidence="43 sealed artifacts", reason="unchanged bytes",
                     verified_by="sealed digests plus spirv-val")],
        limits=["measured geometries only"],
    )
    return receipt, evidence_dir


class Phase16ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.receipt, self.evidence = build(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, receipt=None, sha=None, ancestor=None):
        return verify_receipt(
            receipt if receipt is not None else self.receipt, self.dir,
            source_head=HEAD,
            current_sha256=sha or (lambda name: "a" * 64),
            ancestor_check=ancestor or (lambda bound, head: True),
        )

    def assertRejected(self, receipt=None, sha=None, ancestor=None):
        with self.assertRaises(ValueError):
            self.verify(receipt, sha, ancestor)

    def changed(self, mutation):
        receipt = copy.deepcopy(self.receipt)
        mutation(receipt)
        return receipt

    def test_a_genuine_receipt_passes(self):
        self.verify()

    def test_missing_receipt_is_unqualified_not_a_pass(self):
        import check_phase16_receipts as checker
        code = subprocess.run(
            [sys.executable, str(Path(checker.__file__)),
             "--receipt", str(self.dir / "absent.json"), "--source-head", HEAD],
            capture_output=True, text=True,
        ).returncode
        self.assertEqual(code, 3, CURRENT_RECEIPT.as_posix())

    def test_receipt_bound_to_an_unrelated_revision_is_rejected(self):
        # The ancestry check only runs when the bound revision differs from the
        # checkout, so this receipt must name another revision first.
        receipt = copy.deepcopy(self.receipt)
        receipt["source_head"] = "2" * 40
        with self.assertRaises(ValueError):
            verify_receipt(receipt, self.dir, source_head=HEAD,
                           current_sha256=lambda name: "a" * 64,
                           ancestor_check=lambda bound, head: False)

    def test_changed_transformer_source_is_rejected(self):
        name = "hierarchos-vulkan/src/transformer.rs"

        def sha(path):
            return "b" * 64 if path == name else "a" * 64

        self.assertRejected(sha=sha)

    def test_changed_shader_manifest_is_rejected(self):
        name = "hierarchos-vulkan/validation/training_shader_manifest.json"
        self.assertRejected(sha=lambda path: "b" * 64 if path == name else "a" * 64)

    def test_changed_query_shader_is_rejected(self):
        name = "hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.comp"
        self.assertRejected(sha=lambda path: "b" * 64 if path == name else "a" * 64)

    def test_changed_key_shader_is_rejected(self):
        name = "hierarchos-vulkan/shaders/transformer_attention_backward_key_rdna3.comp"
        self.assertRejected(sha=lambda path: "b" * 64 if path == name else "a" * 64)

    def test_changed_spirv_artifact_is_rejected(self):
        name = "hierarchos-vulkan/shaders/transformer_attention_backward_query_rdna3.spv"
        self.assertRejected(sha=lambda path: "b" * 64 if path == name else "a" * 64)

    def test_missing_gradient_case_is_rejected(self):
        def drop(receipt):
            receipt["attention_backward"]["cases"] = 3
        self.assertRejected(self.changed(drop))

    def test_incomplete_tensor_comparison_is_rejected(self):
        def truncate(receipt):
            receipt["attention_backward"]["bitwise_equal"] -= 1
        self.assertRejected(self.changed(truncate))

    def test_surviving_output_sentinel_is_rejected(self):
        def sentinel(receipt):
            receipt["attention_backward"]["unwritten"] = 1
        self.assertRejected(self.changed(sentinel))

    def test_zero_query_dispatch_is_rejected(self):
        def zero(receipt):
            receipt["attention_backward"]["dispatches"]["twopass_arm"]["query"] = 0
        self.assertRejected(self.changed(zero))

    def test_zero_key_dispatch_is_rejected(self):
        def zero(receipt):
            receipt["attention_backward"]["dispatches"]["twopass_arm"]["key"] = 0
        self.assertRejected(self.changed(zero))

    def test_unexpected_scalar_fallback_is_rejected(self):
        def fallback(receipt):
            receipt["attention_backward"]["dispatches"]["twopass_arm"]["scalar"] = 2
        self.assertRejected(self.changed(fallback))

    def test_incorrect_binary_hash_is_rejected(self):
        def short(receipt):
            receipt["binaries"]["libtest"] = "a" * 40
        self.assertRejected(self.changed(short))

    def test_forged_artifact_count_is_rejected(self):
        def forge(receipt):
            receipt["shaders"]["inventory"] = 44
            receipt["shaders"]["status_counts"] = {"identical": 44}
        self.assertRejected(self.changed(forge))

    def test_changed_historical_artifact_is_rejected(self):
        def swap(receipt):
            receipt["shaders"]["historical"]["rows"][0]["sha256"] = "b" * 64
        self.assertRejected(self.changed(swap))

    def test_phase_xv_receipt_presented_as_current_is_rejected(self):
        def past(receipt):
            receipt["schema"] = 1
            receipt["scope"] = "gpt2_frozen_shape_matrix_and_retained_parity"
        self.assertRejected(self.changed(past))

    def test_tampered_evidence_file_is_rejected(self):
        (self.evidence / "isolation.json").write_text('{"changed": true}\n')
        self.assertRejected()

    def test_tampered_two_pass_log_is_rejected(self):
        path = self.evidence / "two-pass" / "test.log"
        path.write_text(path.read_text().replace("causal=1 masked=false", "causal=0 masked=false", 1))
        self.assertRejected()

    def test_missing_evidence_file_is_rejected(self):
        (self.evidence / "primitives.log").unlink()
        self.assertRejected()

    def test_relaxed_policy_is_rejected(self):
        def relax(receipt):
            receipt["absolute_limit"] = 2e-6
        self.assertRejected(self.changed(relax))

    def test_unbound_new_source_is_rejected(self):
        def drop(receipt):
            receipt["sources"].pop(EXTRA_SOURCES[0])
        self.assertRejected(self.changed(drop))


if __name__ == "__main__":
    unittest.main()

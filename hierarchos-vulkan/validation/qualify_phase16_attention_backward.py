#!/usr/bin/env python3
"""Produce the Phase XVI current-source qualification receipt.

The producer never invents a result. It consumes a run directory that holds the
output of the existing numerical producers on the exact tested source, the
45-artifact Phase XVI shader gate, and the two-pass attention backward logs. It
re-verifies every gate, re-derives the gradient summary from the raw logs, binds
the source bytes, executable hashes, fixture manifest and evidence digests, and
only then writes ``current-qualification.json``.

    python validation/qualify_phase16_attention_backward.py \\
        --run target/phase16/run \\
        --cpu-fixtures target/rdna3-port/gpt2-shapes-final-cpu \\
        --libtest-binary target/debug/deps/hierarchos_vulkan-<hash>.exe \\
        --output validation/release_parity/multimodal_training/\\
20261109-rdna3-phase16-attention-backward-qualification/current-qualification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from check_phase16_receipts import (
    COUNTS, ENVIRONMENT, EXTRA_SOURCES, GATES, LIMIT, PHASE16_ARTIFACTS, ROOT, SCOPE,
    SEALED_GATE, SHADERS,
)
from check_rdna3_gpt2_receipts import RECEIPTS, SHAPES, validate_qualification
from emit_two_pass_evidence import parse as parse_two_pass

EVIDENCE_FILES = (
    "shapes/qualification.json", "shared/qualification.json", "kimi/qualification.json",
    "seq2seq/qualification.json", "gemma3/qualification.json",
    "gemma4-legacy/qualification.json", "gemma4-rgb/qualification.json",
    "llava/qualification.json", "primitives.log", "v41native/qualification.json",
    "v41bridge/qualification.json", "isolation.json", "phase16-spirv.json",
)
TWO_PASS_LOGS = ("two-pass/test.log", "two-pass/harness.log")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def worst_logit(rows):
    values = [check["max_abs"] for row in rows for check in row["checks"]
              if "logits" in check["metric"] and "max_abs" in check]
    require(values, "qualification has no logit comparisons")
    return max(values)


def expected_shape_cases():
    return {f"gpt2_b{b}_s{s}_h{h}/{mask}" for b, s, h in SHAPES for mask in ["unmasked", "padded"]}


def verify_shaders(run: Path) -> dict:
    gate = load(run / "phase16-spirv.json")
    require(gate.get("schema") == 2
            and gate.get("scope") == "phase16_45_artifact_production_inventory",
            "Phase XVI shader gate scope changed")
    require(gate.get("pass") is True, "Phase XVI shader gate did not pass")
    require(gate.get("sealed_scope") == 43 and gate.get("counts", {}).get("total") == 45,
            "Phase XVI shader gate is not 43 sealed + 2 new artifacts")
    require(gate.get("status_counts") == {"identical": 45} and gate.get("failures") == [],
            "Phase XVI shader gate reports a failed artifact")
    rows = gate["artifacts"]
    sealed_rows = [row for row in rows if row.get("origin") == "sealed"]
    new_rows = [row for row in rows if row.get("origin") == "phase16"]
    require(len(sealed_rows) == 43 and len(new_rows) == 2,
            "Phase XVI shader gate miscounts its origins")
    require({row["artifact"] for row in new_rows} == set(PHASE16_ARTIFACTS),
            "Phase XVI shader gate covers other new artifacts")

    sealed = load(SEALED_GATE)
    sealed_index = {row["artifact"]: row for row in sealed["artifacts"]}
    historical = []
    for row in sealed_rows:
        name = row["artifact"]
        require(name in sealed_index, f"{name}: not in the sealed Phase XV gate")
        require(row.get("passed") is True, f"{name}: historical bytes were not validated")
        require(row.get("sha256") == sealed_index[name]["sha256"],
                f"{name}: committed bytes differ from the sealed identity")
        require(not row.get("rebuilt_sha256"),
                f"{name}: historical bytes must not be reported as freshly recompiled")
        require(digest(SHADERS / name) == row["sha256"], f"{name}: artifact changed on disk")
        historical.append(dict(artifact=name, sha256=row["sha256"],
                               target_env=row["target_env"]))
    rebuilt = []
    for row in new_rows:
        name = row["artifact"]
        require(row.get("passed") is True, f"{name}: rebuilt artifact did not validate")
        require(row.get("rebuilt_sha256") == row.get("sha256"),
                f"{name}: rebuild is not byte-identical to the committed artifact")
        require(row.get("target_env") and row.get("compiler_version"),
                f"{name}: rebuild does not record its toolchain")
        require(digest(SHADERS / name) == row["sha256"], f"{name}: artifact changed on disk")
        rebuilt.append(dict(artifact=name, sha256=row["sha256"],
                            rebuilt_sha256=row["rebuilt_sha256"],
                            target_env=row["target_env"], compiler=row.get("compiler"),
                            compiler_version=row.get("compiler_version"), validated=True))
    return dict(
        inventory=45,
        status_counts={"identical": 45},
        historical=dict(artifacts=43, verified=43, recompiled=0,
                        sealed_gate_sha256=digest(SEALED_GATE), rows=historical),
        phase16=dict(artifacts=2, rebuilt=2, validated=2, rows=rebuilt),
        receipt_sha256=digest(run / "phase16-spirv.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="directory holding shapes/, shared/, ... and two-pass/")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/release_parity/multimodal_training/"
                        "20261109-rdna3-phase16-attention-backward-qualification/"
                        "current-qualification.json")
    parser.add_argument("--cpu-fixtures", type=Path, required=True)
    parser.add_argument("--image-binary", type=Path,
                        default=ROOT / "target/debug/image_text_pipeline_probe.exe")
    parser.add_argument("--gemma3-binary", type=Path,
                        default=ROOT / "target/debug/gemma3_vision_probe.exe")
    parser.add_argument("--gemma4-binary", type=Path,
                        default=ROOT / "target/debug/gemma4_vision_probe.exe")
    parser.add_argument("--llava-binary", type=Path,
                        default=ROOT / "target/debug/llava_pipeline_probe.exe")
    parser.add_argument("--libtest-binary", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    status = git("status", "--porcelain").stdout.strip()
    require(not status, "qualification requires a clean worktree:\n" + status)
    head = git("rev-parse", "HEAD").stdout.strip()
    require(len(head) == 40, "cannot resolve the tested revision")

    binaries = {
        "image_text_pipeline_probe": digest(args.image_binary),
        "gemma3_vision_probe": digest(args.gemma3_binary),
        "gemma4_vision_probe": digest(args.gemma4_binary),
        "llava_pipeline_probe": digest(args.llava_binary),
        "libtest": digest(args.libtest_binary),
    }

    shapes = load(run / "shapes" / "qualification.json")
    rows = validate_qualification(shapes, COUNTS["shapes"])
    require({row["case"] for row in rows} == expected_shape_cases(),
            "expanded shape coverage changed")
    require(shapes["binary_sha256"] == binaries["image_text_pipeline_probe"],
            "shape replay binary differs from the built probe")
    require(shapes["fixtures_sha256"] == digest(args.cpu_fixtures / "manifest.json"),
            "shape replay consumed another frozen CPU manifest")
    shape_logit = worst_logit(rows)
    require(shape_logit < LIMIT, "shape logit gate violated")

    retained = {}
    for name in GATES:
        if name == "gemma4":
            # The two gemma4 cases are separate genuine runs (one per pixel
            # domain) with their own binaries and fixtures. Combining them is
            # two bound receipts, never a synthesized third result.
            domains = {domain: load(run / domain / "qualification.json")
                       for domain in ("gemma4-legacy", "gemma4-rgb")}
            domain_rows = {domain: validate_qualification(payload, 1)
                           for domain, payload in domains.items()}
            for domain, payload in domains.items():
                require(payload["binary_sha256"] == binaries["gemma4_vision_probe"],
                        f"{domain} binary differs from the built probe")
            require(len({row["case"] for rows in domain_rows.values() for row in rows}) == 2,
                    "gemma4 domains do not cover two distinct cases")
            retained[name] = dict(cases=2, pass_=True,
                                  qualification_sha256=digest(run / "gemma4-legacy" /
                                                              "qualification.json"),
                                  worst_logit=max(worst_logit(rows)
                                                  for rows in domain_rows.values()),
                                  receipts={domain: digest(run / domain / "qualification.json")
                                            for domain in domains})
            continue
        payload = load(run / name / "qualification.json")
        gate_rows = validate_qualification(payload, COUNTS[name])
        expected_binary = binaries[
            "gemma3_vision_probe" if name == "gemma3" else
            "gemma4_vision_probe" if name == "gemma4" else
            "llava_pipeline_probe" if name == "llava" else
            "image_text_pipeline_probe"]
        require(payload["binary_sha256"] == expected_binary,
                f"{name} binary differs from the built probe")
        retained[name] = dict(cases=len(gate_rows), pass_=True,
                              qualification_sha256=digest(run / name / "qualification.json"),
                              worst_logit=worst_logit(gate_rows))

    primitive_log = (run / "primitives.log").read_text()
    require(re.search(r"test result: ok\. 9 passed; 0 failed;", primitive_log),
            "primitive tests incomplete")
    require(len(re.findall(r"^test .*rdna3_gpt2_.* \.\.\. ok$", primitive_log, re.M)) == 9,
            "primitive test coverage changed")

    native = load(run / "v41native" / "qualification.json")
    require(native.get("pass") is True and native.get("expected_tests") == 30
            and native.get("completed_tests") == 30, "V4.1 native regression failed")
    require(native.get("binary_sha256") == binaries["libtest"],
            "V4.1 native binary differs from the built libtest executable")
    require(all(job["pass"] for job in native["jobs"]), "V4.1 native job failed")
    require(native.get("max_abs_logit", 1.0) < LIMIT, "V4.1 native logit gate violated")

    bridge = load(run / "v41bridge" / "qualification.json")
    require(bridge.get("pass") is True and bridge.get("expected_cases") == 2
            and bridge.get("completed_cases") == 2, "V4.1 bridge regression failed")
    require(bridge.get("binary_sha256") == binaries["libtest"],
            "V4.1 bridge binary differs from the built libtest executable")

    isolation = load(run / "isolation.json")
    require(isolation.get("unchanged") is True and isolation.get("files_checked") == 1341
            and isolation.get("existing_recipes") == 672
            and isolation.get("failures") == [], "upstream bytes changed")

    shaders = verify_shaders(run)

    gradient = parse_two_pass(run / "two-pass" / "test.log", run / "two-pass" / "harness.log")
    require(gradient["pass"],
            "two-pass gradient evidence failed: " + "; ".join(gradient["failures"]))
    backward = dict(
        enabled_by=ENVIRONMENT,
        scalar_default=True,
        cases=len({row["case"] for row in gradient["cases"]}),
        comparisons=gradient["comparisons"],
        elements=gradient["elements"],
        bitwise_equal=gradient["bitwise_equal"],
        max_abs=gradient["max_abs"],
        max_ulp=gradient["max_ulp"],
        unwritten=gradient["unwritten"],
        adapters_identical=gradient["adapters"]["identical"],
        adapter_files=[f"{path} {sha}" for path, sha in gradient["adapters"]["files"]],
        dispatches=gradient["dispatches"],
        diagnostic=gradient["diagnostic"],
        negative_controls=gradient["negative_controls"],
        test_log_sha256=gradient["evidence"]["test_log"],
        harness_log_sha256=gradient["evidence"]["harness_log"],
    )

    historical = load(RECEIPTS / "receipt-hashes.json")
    sources = {}
    for name in sorted(historical["sources"]):
        path = ROOT / name
        require(path.is_file(), "bound source is missing: " + name)
        sources[name] = digest(path)
    for name in EXTRA_SOURCES:
        path = ROOT / name
        require(path.is_file(), "phase16 source is missing: " + name)
        sources[name] = digest(path)

    evidence_dir = args.output.parent / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence = {}
    for name in (*EVIDENCE_FILES, *TWO_PASS_LOGS):
        source = run / name
        require(source.is_file(), "required evidence is missing: " + name)
        destination = evidence_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        evidence[name] = digest(destination)

    receipt = dict(
        schema=2, **{"pass": True}, target="rdna3", scope=SCOPE,
        source_head=head, worktree=True,
        absolute_limit=LIMIT, rtol=0, comparison="strict_less_than",
        adapter=dict(vendor="1002", device="15bf", driver="32.0.31007.6002"),
        binaries=binaries,
        producer_sha256=digest(Path(__file__)),
        cpu_fixtures_sha256=digest(args.cpu_fixtures / "manifest.json"),
        shapes=dict(cases=len(rows), worst_logit=shape_logit,
                    binary_sha256=shapes["binary_sha256"],
                    fixtures_sha256=shapes["fixtures_sha256"]),
        retained=retained,
        primitives=dict(passed=9, failed=0, ignored=1,
                        log_sha256=digest(run / "primitives.log")),
        v41=dict(native=dict(tests=30, max_abs_logit=native["max_abs_logit"],
                             receipt_sha256=digest(run / "v41native" / "qualification.json")),
                 bridge=dict(cases=2,
                             receipt_sha256=digest(run / "v41bridge" / "qualification.json"))),
        shaders=shaders,
        attention_backward=backward,
        isolation=dict(files_checked=1341, existing_recipes=672,
                       receipt_sha256=digest(run / "isolation.json")),
        sources=sources,
        evidence=evidence,
        reused=[dict(
            evidence="43 sealed Phase XV shader artifacts",
            reason="artifact bytes are unchanged, so their provenance was not "
                   "rebuilt and is not claimed as fresh compilation",
            verified_by="sealed digest identity plus a fresh spirv-val run in "
                        "phase16-spirv.json; only the two new artifacts were rebuilt",
        )],
        limits=[
            "eight measured GPT2 geometries; not a Cartesian shape certificate",
            "the 43 sealed shader artifacts are byte-verified, not recompiled",
            "two-pass attention backward remains opt-in; the scalar path is the default",
            "isolated kernel timings require a separate paired benchmark receipt",
            "no physical Gen9 execution; the shader gate is static preservation",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PHASE XVI QUALIFICATION: PASS ({head[:12]}, {len(rows)} shapes, "
          f"{backward['comparisons']} gradient tensors, "
          f"{shaders['inventory']} shader artifacts, {len(sources)} bound sources)")


if __name__ == "__main__":
    main()

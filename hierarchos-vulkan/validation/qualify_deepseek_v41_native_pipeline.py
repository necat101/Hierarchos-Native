"""Qualify native V4.1 image training, reload and preserved text controls.

Pass an already built Rust libtest executable. This command runs on the local
Vulkan device; it never builds, downloads an oracle or contacts another host.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

JOBS = [
    ("primitives", "deepseek_v41_native_avx2_", 12),
    ("vision", "deepseek_v41_native_vision_", 4),
    ("joint", "deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu", 1),
    ("checkpoint", "deepseek_v41_native_joint_checkpoint_resume_export_and_freshness", 1),
    ("modality", "deepseek_v41_modality", 4),
    ("text", "deepseek_v41_full_model_cpu_autograd_adamw_resume_and_export_gate", 1),
    ("mathscope", "native_avx2_scope_preserves_host_dispatch", 1),
    ("mathscope-wide-host", "native_avx2_scope_preserves_host_dispatch", 1),
    ("vision-wide-host", "deepseek_v41_native_vision_", 4),
    ("joint-wide-host", "deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu", 1),
]
TOLERANCE = 2e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, filter_, *flags, env=None):
    return subprocess.run([str(binary), filter_, *flags], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def verify(log, names):
    summary = re.search(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", log)
    if summary is None or tuple(map(int, summary.groups())) != (len(names), 0, 0):
        return False
    # Check identities as well as counts. Device messages may occur between
    # the libtest prefix and its trailing `ok` when --nocapture is enabled.
    return all(re.search(r"^test " + re.escape(name) + r" \.\.\.", log, re.M) for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["gen9", "rdna3", "local"], default="local")
    parser.add_argument("--device-id", type=lambda x: int(x, 0))
    parser.add_argument("--source-manifest", type=Path,
                        help="Optional hash manifest emitted by the build; checked against this workspace")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    # The maintained copy lives in hierarchos-vulkan/validation.
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_sha = sha(binary)
    sources = {}
    if args.source_manifest:
        sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if not sources or any(sha(root / p) != h for p, h in sources.items()):
            raise ValueError("Build source manifest differs from the workspace")
        (output / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    fixtures = root / "hierarchos-vulkan/validation/fixtures"
    oracle_hashes = {str(p.relative_to(root)): sha(p) for p in [
        fixtures / "deepseek_v41_native_pipeline_avx2.json",
        fixtures / "deepseek_v41_native_vision_avx2.json",
        *sorted((fixtures / "vision_avx2_row_tail").glob("*.json")),
    ]}
    status = []
    for label, filter_, count in JOBS:
        listed = run(binary, filter_, "--list")
        names = re.findall(r"^(\S+): test$", listed.stdout, re.M)
        if listed.returncode or len(names) != count or len(set(names)) != count:
            raise ValueError(f"{label}: expected {count} distinct tests, found {len(names)}")
        wide_host = label.endswith("-wide-host")
        env = dict(os.environ, HIERARCHOS_V41_JOINT_QUALIFICATION=str(output / ("joint-wide-host.json" if wide_host else "joint.json")))
        if wide_host:
            env["HIERARCHOS_ATEN_VECTOR_WIDTH"] = "16"
        print("Starting", label, flush=True)
        result = run(binary, filter_, "--test-threads=1", "--nocapture", env=env)
        log = result.stdout + result.stderr
        (output / (label + ".log")).write_text(log, encoding="utf-8")
        passed = result.returncode == 0 and verify(log, names)
        if sha(binary) != binary_sha:
            raise ValueError("Test executable changed during qualification")
        status.append(dict(name=label, tests=names, expected_tests=count,
                           returncode=result.returncode, **{"pass": passed}))
        (output / "run-status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(label, "PASS" if passed else "FAIL", flush=True)
    joint_path = output / "joint.json"
    joint = json.loads(joint_path.read_text()) if joint_path.exists() else {}
    joint_valid = (joint.get("pass") is True and joint.get("expected_cases") == 2
                   and joint.get("completed_cases") == 2 and joint.get("optimizer_updates") == 4
                   and joint.get("logit_values") == 1360 and joint.get("failures") == []
                   and joint.get("all_comparisons", {}).get("comparisons") == 1866
                   and joint.get("all_comparisons", {}).get("values") == 155444)
    log = (output / "joint.log").read_text(encoding="utf-8")
    metrics = re.findall(r"^V4\.1 full training ([^\r\n]+/(?:inference|training)-logits): (\d+) values, worst (\S+) at", log, re.M)
    max_logit = max((float(m[2]) for m in metrics), default=math.inf)
    logits_valid = (len(metrics) == 10 and sum(int(m[1]) for m in metrics) == 1360
                    and len({m[0] for m in metrics}) == 10
                    and all(math.isfinite(float(m[2])) and float(m[2]) < TOLERANCE for m in metrics))
    if sources and any(sha(root / p) != h for p, h in sources.items()):
        raise ValueError("Workspace source changed during qualification")
    if any(sha(root / p) != h for p, h in oracle_hashes.items()):
        raise ValueError("Frozen CPU oracle changed during qualification")
    passed = all(j["pass"] for j in status) and joint_valid and logits_valid
    limits = ["one complete image span per prefill", "fixed patch geometry per trainer",
              "production geometry and BF16 training unqualified"]
    limits.append("physical RDNA3 replay recorded on the local default Vulkan adapter"
                  if args.target == "rdna3" else "physical RDNA3 replay pending")
    receipt = dict(schema=1, **{"pass": passed}, profile_qualified=passed,
                   architecture="deepseek_v41", contract="fp32_native_vision_frozen_language_lora",
                   cpu_math="windows_aten_intel_avx2_v1", tolerance=TOLERANCE, rtol=0,
                   comparison="strict_less_than", target=args.target, device_id=args.device_id,
                   device_selection="local default Vulkan adapter; target supplied by the operator",
                   expected_tests=30,
                   completed_tests=sum(j["expected_tests"] for j in status if j["pass"]),
                   native_cases=2, optimizer_updates=4, logit_values=1360,
                   max_abs_logit=max_logit if math.isfinite(max_logit) else None,
                   joint_comparisons=joint.get("all_comparisons"),
                   checkpoint_resume_export_verified=next(j["pass"] for j in status if j["name"] == "checkpoint"),
                   binary_sha256=binary_sha, oracle_sha256=oracle_hashes,
                   source_manifest_sha256=sha(output / "source-sha256.json") if sources else None,
                   jobs=status, limits=limits)
    (output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Native V4.1 qualification", "PASS" if passed else "FAIL", "max_abs_logit", max_logit)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

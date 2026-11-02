"""Paired release probes with frozen Gen9 digest guards.

Run on an idle adapter after builds/tests finish. This measures the retained
scaled checkpoint; final-row digests are regression guards, not full CPU-logit
qualification. The strict Rust gates remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

FROZEN = {("resident", 8, 8): "90e096ebc5bf0659",
          ("default", 8, 4): "2d2f8e2d5914a0d2",
          ("resident", 1, 1): "8e61e9bd28665ff1",
          ("resident", 2, 1): "95e693dcb9e76429",
          ("resident", 3, 1): "17cc1ba6404ff8d6"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.pairs < 2 or args.steps < 1:
        parser.error("at least two alternating pairs and one measured step are required")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    records = []

    def run(arm, mode, prefill, decode, suffix, *, steps, warmup, staged=False, profile=False):
        env = os.environ.copy()
        for name in list(env):
            if (name.startswith("HIERARCHOS_VULKAN_PROFILE_")
                or name in {"HIERARCHOS_VULKAN_LAYER_DIGEST", "HIERARCHOS_VULKAN_DIGEST_ONLY",
                            "HIERARCHOS_VULKAN_SUBMIT_TRACE", "HIERARCHOS_VULKAN_MOE_TRACE",
                            "HIERARCHOS_VULKAN_V41_FUSED_EXPERTS", "HIERARCHOS_VULKAN_LAYER_SESSION"}):
                env.pop(name)
        if staged:
            env["HIERARCHOS_VULKAN_V41_FUSED_EXPERTS"] = "0"
        if profile:
            env.update(HIERARCHOS_VULKAN_PROFILE_KERNELS="1",
                       HIERARCHOS_VULKAN_PROFILE_SAMPLE_STRIDE="1",
                       HIERARCHOS_VULKAN_PROFILE_TOP_SHADERS="4096")
        command = [str(binaries[arm]), "--residency", mode, "--prefill", str(prefill),
                   "--decode", str(decode), "--steps", str(steps), "--warmup", str(warmup)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        output = result.stdout + result.stderr
        log = f"{arm}-{mode}-{prefill}-{decode}-{suffix}.log"
        (args.output / log).write_text(output, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"probe failed ({result.returncode}); see {log}")
        digests = re.findall(r"v41_probe_logits .*?digest=([0-9a-f]+)", output)
        expected = FROZEN[mode, prefill, decode]
        if len(digests) < steps or any(d != expected for d in digests):
            raise RuntimeError(f"{log}: frozen digest mismatch {digests}, expected {expected}")
        phases = []
        for line in output.splitlines():
            if line.startswith("v41_probe_phase ") and "measured=true" in line:
                fields = dict(token.split("=", 1) for token in line.split()[1:])
                phases.append({"phase": fields["phase"], "ms": float(fields["ms"]),
                               "submissions": int(fields["submissions"])})
        if sum(p["phase"] == "prefill" for p in phases) != steps:
            raise RuntimeError(f"{log}: incomplete prefill measurements")
        if sum(p["phase"] == "decode" for p in phases) != steps * decode:
            raise RuntimeError(f"{log}: incomplete decode measurements")
        record = dict(arm=arm, residency=mode, prefill=prefill, decode=decode, log=log,
                      staged=staged, profile=profile, digest=expected, phases=phases,
                      context=[line for line in output.splitlines()
                               if line.startswith(("v41_probe_load ", "v41_probe_io "))])
        records.append(record)
        print("PASS", log, expected, flush=True)

    for mode, decode in [("resident", 8), ("default", 4)]:
        for pair in range(args.pairs):
            for arm in (["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]):
                run(arm, mode, 8, decode, f"pair{pair}", steps=args.steps, warmup=1)
    for prefill in (1, 2, 3):
        for arm in binaries:
            run(arm, "resident", prefill, 1, "guard", steps=1, warmup=0)
    run("candidate", "resident", 8, 8, "staged-guard", steps=1, warmup=0, staged=True)
    if args.profile:
        for arm in binaries:
            run(arm, "resident", 8, 8, "profile", steps=1, warmup=0, profile=True)

    timings = {}
    for mode in ("resident", "default"):
        timings[mode] = {}
        for arm in binaries:
            phases = [p for r in records if r["arm"] == arm and r["residency"] == mode
                      and "-pair" in r["log"] for p in r["phases"]]
            timings[mode][arm] = {
                f"{phase}_ms": statistics.median(p["ms"] for p in phases if p["phase"] == phase)
                for phase in ("prefill", "decode")}
        timings[mode]["latency_reduction_percent"] = {
            phase: 100 * (1 - timings[mode]["candidate"][phase] / timings[mode]["baseline"][phase])
            for phase in ("prefill_ms", "decode_ms")}
    report = dict(binaries={k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                            for k, v in binaries.items()}, timings=timings, runs=records,
                  limits=["Scaled checkpoint", "Final-row digest guards, separate full-logit Rust gates",
                          "Unprofiled paired phases only contribute to latency medians"])
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase XVI: qualify the 45-artifact production shader inventory.

Phase XV sealed 43 artifacts. The two-pass attention backward adds two more
production artifacts, so the current inventory is 45 and the Phase XV
qualification cannot describe it. This producer is a separate, versioned gate;
the Phase XV receipt and its 43-artifact inventory are read but never written.

For the 43 sealed artifacts it reuses the sealed identities only (name, digest,
target environment) and re-executes the validator on the bytes in this checkout,
so a changed artifact fails instead of inheriting the old result. For the two new
artifacts it rebuilds each one from the manifest's own command and requires the
bytes to match the committed artifact, then validates the committed bytes.

    python validation/emit_phase16_spirv_gate.py \
        --output target/phase16/current/phase16-spirv.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEALED = (
    ROOT / "validation" / "release_parity" / "multimodal_training"
    / "20261108-rdna3-phase15-qualification" / "gates" / "spirv.json"
)
MANIFEST = ROOT / "validation" / "training_shader_manifest.json"
SHADERS = ROOT / "shaders"
PHASE16_SCOPE = (
    "transformer_attention_backward_query_rdna3.spv",
    "transformer_attention_backward_key_rdna3.spv",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_validator(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    for name in ("spirv-val.exe", "spirv-val"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for candidate in sorted(Path("C:/VulkanSDK").glob("*/Bin/spirv-val.exe"), reverse=True):
        return candidate
    raise SystemExit("spirv-val is not installed; the gate cannot be produced")


def compiler_version(binary: Path) -> str:
    run = subprocess.run([str(binary), "--version"], capture_output=True, text=True)
    require(run.returncode == 0, f"{binary} --version failed")
    return (run.stdout + run.stderr).strip().splitlines()[0]


def manifest_entry(entries: dict, artifact: str) -> dict:
    require(artifact in entries, f"{artifact} has no manifest entry")
    return entries[artifact]


def rebuild_command(entry: dict) -> list[str]:
    flags = list(entry["flags"])
    compiler = entry["compiler"]
    if compiler == "glslc":
        require(shutil.which("glslc.exe") or Path("C:/VulkanSDK/1.4.321.1/Bin/glslc.exe").is_file(),
                "glslc is not installed")
        binary = Path(shutil.which("glslc.exe") or "C:/VulkanSDK/1.4.321.1/Bin/glslc.exe")
    elif compiler == "glslangValidator":
        found = shutil.which("glslangValidator.exe")
        require(found is not None, "glslangValidator is not installed")
        binary = Path(found)
    else:
        raise ValueError(f"unknown manifest compiler {compiler}")
    reported = compiler_version(binary)
    if entry.get("compiler_version"):
        require(
            entry["compiler_version"] in reported,
            f"{entry['artifact']}: manifest pins {compiler} {entry['compiler_version']} "
            f"but the installed binary reports {reported}",
        )
    return [str(binary), *flags, str(SHADERS / entry["source"]), "-o"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, default=SEALED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator", type=Path)
    args = parser.parse_args()

    validator = find_validator(args.validator)
    validator_version = compiler_version(validator)
    entries = {row["artifact"]: row for row in json.loads(args.manifest.read_text())["shaders"]}
    sealed = json.loads(args.sealed.read_text())
    sealed_rows = sealed["artifacts"]
    require(len(sealed_rows) == 43, f"sealed inventory is {len(sealed_rows)}, expected 43")

    records, failures = [], []
    for row in sealed_rows:
        name = row["artifact"]
        path = SHADERS / name
        if not path.is_file():
            records.append(dict(artifact=name, origin="sealed", passed=False,
                                diagnostic="missing from this checkout"))
            failures.append(f"{name}: missing")
            continue
        sha = digest(path)
        if sha != row["sha256"]:
            records.append(dict(artifact=name, origin="sealed", sha256=sha, passed=False,
                                diagnostic="committed bytes differ from the sealed artifact"))
            failures.append(f"{name}: bytes changed")
            continue
        run = subprocess.run([str(validator), "--target-env", row["target_env"], str(path)],
                             capture_output=True, text=True)
        passed = run.returncode == 0
        records.append(dict(artifact=name, origin="sealed", sha256=sha,
                            target_env=row["target_env"], passed=passed,
                            diagnostic=(run.stdout + run.stderr).strip()))
        if not passed:
            failures.append(f"{name}: spirv-val exit {run.returncode}")

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for name in PHASE16_SCOPE:
            entry = manifest_entry(entries, name)
            path = SHADERS / name
            if not path.is_file():
                records.append(dict(artifact=name, origin="phase16", passed=False,
                                    diagnostic="missing from this checkout"))
                failures.append(f"{name}: missing")
                continue
            sha = digest(path)
            rebuilt = temporary / name
            command = rebuild_command(entry)
            run = subprocess.run([*command, str(rebuilt)],
                                 capture_output=True, text=True, cwd=ROOT)
            if run.returncode != 0:
                records.append(dict(artifact=name, origin="phase16", sha256=sha, passed=False,
                                    diagnostic=f"rebuild failed: {(run.stdout + run.stderr).strip()}"))
                failures.append(f"{name}: rebuild failed")
                continue
            rebuilt_sha = digest(rebuilt)
            if rebuilt_sha != sha:
                records.append(dict(artifact=name, origin="phase16", sha256=sha,
                                    rebuilt_sha256=rebuilt_sha, passed=False,
                                    diagnostic="rebuilt bytes differ from the committed artifact"))
                failures.append(f"{name}: rebuild is not byte-identical")
                continue
            target_env = entry["target_env"]
            validate = subprocess.run([str(validator), "--target-env", target_env, str(path)],
                                      capture_output=True, text=True)
            passed = validate.returncode == 0
            records.append(dict(artifact=name, origin="phase16", sha256=sha,
                                rebuilt_sha256=rebuilt_sha, target_env=target_env,
                                compiler=entry["compiler"],
                                compiler_version=entry["compiler_version"],
                                passed=passed,
                                diagnostic=(validate.stdout + validate.stderr).strip()))
            if not passed:
                failures.append(f"{name}: spirv-val exit {validate.returncode}")

    artifacts = [row["artifact"] for row in records]
    require(len(artifacts) == len(set(artifacts)), "artifact appears twice")
    status_counts = {
        "identical": sum(1 for row in records if row["passed"]),
        "failed": len(failures),
    }
    report = dict(
        schema=2,
        scope="phase16_45_artifact_production_inventory",
        **{"pass": not failures},
        validator=validator_version,
        sealed_scope=len(sealed_rows),
        phase16_scope=list(PHASE16_SCOPE),
        counts=dict(total=len(records), **status_counts),
        status_counts=status_counts,
        artifacts=records,
        failures=failures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in records:
        print(f"{'identical' if row['passed'] else 'FAILED':10} {row['origin']:9} {row['artifact']}")
    print(f"artifacts {len(records)} (= {len(sealed_rows)} sealed + {len(PHASE16_SCOPE)} phase16), "
          f"validator {validator_version}")
    if failures:
        print(f"FAIL: {len(failures)} artifact(s) did not qualify")
        return 1
    print("PASS: every production artifact is byte-identical and validates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rebuild and validate every shader covered by the training optimization release.

The manifest records the actual compiler, that compiler's release and the flags
for each embedded variant; attention, reductions and mixed-precision variants do
not share one build recipe, and a tree rebuilt across two compiler releases can
pin a release per artifact. Every recipe is therefore compiled by the candidate
that reports the release it records (`--compiler-path NAME=PATH` replaces PATH for
one name), which checks a spanned tree instead of reporting its untranslated
artifacts as drift. Compilation goes to temporary files and never overwrites the
shipping artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import struct
import tempfile

from check_shader_provenance import choose, discover_local, gather, parse_compiler_paths

CRATE = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("training_shader_manifest.json")


def shader_storage_usage(binary):
    """Read the fixed workgroup ABI and explicit Workgroup variable count."""
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    offset, variables, local_size = 5, 0, None
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        if count == 0 or offset + count > len(words):
            raise RuntimeError("malformed SPIR-V instruction")
        if opcode == 59 and count >= 4 and words[offset + 3] == 4:
            variables += 1
        if opcode == 16 and count == 6 and words[offset + 2] == 17:
            local_size = list(words[offset + 3:offset + 6])
        offset += count
    return {"workgroup_variables": variables, "local_size": local_size}


def pinned_toolchain_problem(recipe, decision):
    """Explain why no installed candidate reports an artifact's recorded release."""
    observed = ", ".join(str(item) for item in decision.get("observed_versions", [])) or "nothing"
    return (
        f"{recipe['compiler']} reports {observed} but {recipe['artifact']} records "
        f"{decision['expected_version']}; install that release or pass "
        "--compiler-path NAME=PATH"
    )


def verify(manifest=MANIFEST, overrides=None):
    recipes = json.loads(Path(manifest).read_text(encoding="utf-8"))["shaders"]
    validator = shutil.which("spirv-val")
    if validator is None:
        raise RuntimeError("spirv-val must be on PATH")
    toolchains = gather(sorted({recipe["compiler"] for recipe in recipes}), overrides or {})
    rows = []
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="training-shader-release-") as scratch:
        for recipe in recipes:
            toolchain, decision = choose(recipe, toolchains.get(recipe["compiler"], []))
            if toolchain is None:
                if decision["status"] == "missing_compiler":
                    raise RuntimeError(f"{recipe['compiler']} must be on PATH")
                rows.append({**recipe, "compile_exit": None, "matches": False, "valid": False,
                             "policy_matches": True, "toolchain": decision["status"],
                             "error": pinned_toolchain_problem(recipe, decision)})
                continue
            compiler = toolchain.path
            source = CRATE / "shaders" / recipe["source"]
            embedded = CRATE / "shaders" / recipe["artifact"]
            rebuilt = Path(scratch) / embedded.name
            command = [compiler, *recipe["flags"], str(source), "-o", str(rebuilt)]
            result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True)
            row = {**recipe, "source_sha256": digest(source), "sha256": digest(embedded),
                   "compile_exit": result.returncode, "matches": False, "valid": False,
                   "policy_matches": True, "toolchain_version": toolchain.version}
            if result.returncode == 0:
                validation = subprocess.run(
                    [validator, "--target-env", recipe["target_env"], str(rebuilt)],
                    text=True, capture_output=True,
                )
                row.update(rebuilt_sha256=digest(rebuilt),
                           matches=rebuilt.read_bytes() == embedded.read_bytes(),
                           valid=validation.returncode == 0)
                policy = recipe.get("storage_policy")
                if validation.returncode == 0 and policy:
                    usage = shader_storage_usage(rebuilt.read_bytes())
                    row["storage_usage"] = usage
                    row["policy_matches"] = (
                        usage["local_size"] == policy["local_size"]
                        and usage["workgroup_variables"] <= policy["max_workgroup_variables"]
                    )
                    if not row["policy_matches"]:
                        row["error"] = f"storage policy {policy} does not match {usage}"
                if validation.returncode:
                    row["error"] = validation.stdout + validation.stderr
            else:
                row["error"] = result.stdout + result.stderr
            rows.append(row)
    return {"pass": all(row["matches"] and row["valid"] and row["policy_matches"] for row in rows), "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--compiler-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a candidate binary for a manifest compiler name (repeatable; replaces PATH for that name)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overrides = parse_compiler_paths(args.compiler_path, parser.error)
    # Installed releases are candidates too; the manifest names which one each
    # artifact records, so a tree spanning two releases needs no extra flags.
    root = Path(__file__).resolve().parents[1]
    for name, paths in discover_local(root).items():
        known = overrides.setdefault(name, [])
        known.extend(path for path in paths if path not in known)
    report = verify(args.manifest, overrides)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["variants"]:
        if not row["matches"] or not row["valid"] or not row["policy_matches"]:
            print(f"FAIL {row['artifact']}: matches={row['matches']} valid={row['valid']} "
                  + row.get("error", ""))
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {len(report['variants'])} training shader variants")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

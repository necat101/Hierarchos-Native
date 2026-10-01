"""Verify (or explicitly rebuild) only the Hierarchos bounded ROSA lane shaders."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace embedded artifacts after compilation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiler = shutil.which("glslc")
    if compiler is None:
        parser.error("glslc must be on PATH")
    shader_dir = Path(__file__).resolve().parents[1] / "shaders"
    stem = "rosa_predict_bounded_lanes"
    source = shader_dir / f"{stem}.comp"
    variants = [("", []), *[(f"_subgroup_{width}", [f"ROSA_WORKGROUP_SIZE={width}",
                 "ROSA_USE_SUBGROUP_REDUCTION=1"]) for width in (32, 64, 128, 256)]]
    variants.extend([
        ("_subgroup_256_single_pair", ["ROSA_WORKGROUP_SIZE=256", "ROSA_USE_SUBGROUP_REDUCTION=1", "ROSA_SINGLE_PACKED_PAIR_PER_LANE=1"]),
        ("_subgroup_128_cache_tiled", ["ROSA_WORKGROUP_SIZE=128", "ROSA_USE_SUBGROUP_REDUCTION=1", "ROSA_CACHE_HISTORY_STATE=1"]),
        ("_subgroup_256_single_pair_cache_tiled", ["ROSA_WORKGROUP_SIZE=256", "ROSA_USE_SUBGROUP_REDUCTION=1", "ROSA_SINGLE_PACKED_PAIR_PER_LANE=1", "ROSA_CACHE_HISTORY_STATE=1"]),
    ])
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    with tempfile.TemporaryDirectory(prefix="hierarchos-rosa-shaders-") as scratch:
        for suffix, defines in variants:
            name = stem + suffix + ".spv"
            rebuilt = Path(scratch) / name
            flags = ["--target-env=vulkan1.1", "-O", *["-D" + value for value in defines]]
            subprocess.run([compiler, *flags, str(source), "-o", str(rebuilt)], check=True)
            embedded = shader_dir / name
            if args.write:
                shutil.copyfile(rebuilt, embedded)
            rows.append({"artifact": name, "flags": flags, "sha256": digest(embedded),
                         "matches": digest(embedded) == digest(rebuilt)})
    report = {"compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
              "source_sha256": digest(source), "variants": rows,
              "pass": all(row["matches"] for row in rows)}
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

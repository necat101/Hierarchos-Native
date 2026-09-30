"""Check that selected embedded SPIR-V files reproduce from their GLSL sources.

Compiles into temporary scratch only; never overwrites a production shader.
This detects stale artifacts, but does not qualify numerical model parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shaders", nargs="+", help="shader stems under shaders/")
    parser.add_argument("--compiler", default="glslc")
    parser.add_argument("--compiler-flag", action="append", default=[],
                        help="repeatable compiler flag (use --compiler-flag=-O)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error(f"shader compiler not found: {args.compiler}")
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    rows = []
    with tempfile.TemporaryDirectory(prefix="shader-provenance-") as scratch:
        for stem in args.shaders:
            if Path(stem).name != stem or not stem.replace("_", "").isalnum():
                parser.error(f"expected a shader stem, got {stem!r}")
            source = ROOT / "shaders" / (stem + ".comp")
            embedded = source.with_suffix(".spv")
            rebuilt = Path(scratch) / (stem + ".spv")
            result = subprocess.run([compiler, *args.compiler_flag, str(source), "-o", str(rebuilt)],
                                    text=True, capture_output=True)
            row = {"shader": stem, "source_sha256": sha256(source),
                   "embedded_sha256": sha256(embedded) if embedded.is_file() else None,
                   "compile_exit": result.returncode}
            if result.returncode:
                row.update(matches=False, error=result.stdout + result.stderr)
            else:
                row["rebuilt_sha256"] = sha256(rebuilt)
                row["matches"] = row["rebuilt_sha256"] == row["embedded_sha256"]
            rows.append(row)
    report = {"compiler": compiler, "version": version, "flags": args.compiler_flag, "shaders": rows,
              "pass": all(row["matches"] for row in rows)}
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

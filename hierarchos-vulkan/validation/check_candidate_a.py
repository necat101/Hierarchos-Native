#!/usr/bin/env python3
"""Current-PR gate for the Phase LXII RDNA3 WMMA Candidate A change.

The receipts under ``validation/release_parity`` are historical: each is verified
against the revision it declares, by ``source_provenance``. This gate asks the
other question - does the tree under review still carry the Candidate A
implementation that was qualified, and nothing else on the arithmetic path?

It verifies:

* the four shader artifacts are byte-identical to the Phase LXII qualification;
* both ``.comp`` sources still stage operands as packed 32-bit ``f16vec2`` words
  with the masked 16-bit fallback, and keep exactly four cooperative
  multiply-adds and four fragment loads;
* both ``.spv`` are well-formed SPIR-V 1.4 modules containing exactly four
  ``OpCooperativeMatrixMulAddKHR`` instructions;
* the recorded reproduction receipt binds those exact bytes to their sources;
* the 16,000,000-MAC profitability floor and its dispatch predicate are unchanged;
* inside ``wmma.rs`` the only changed lines are the boundary shape list and its
  comment, and inside ``wmma_bench.rs`` the only removed line is the help string,
  with no timing or statistic statement touched;
* no protected source other than the six Candidate A paths changed, no dependency
  manifest changed, and no model weight or prebuilt executable was added;
* the sample-export flag, the paired ABBA runner and the analyser are present and
  agree, and both newly added boundary shapes are still exercised.

``--reproduce`` additionally recompiles both shaders with the recorded command and
requires byte identity; that needs the shaderc toolchain the artifacts were built
with. It is never claimed for a hosted runner, and this gate never claims a GPU
measurement: the performance record is the Phase LXII evidence measured on the
actual RDNA3 hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "fixtures" / "wmma_candidate_a_reproduction.json"

# The exact base the Phase LXII record was taken from.
BASE_REVISION = "81ac82fa972d84025129e7c3f0e58fdece7df786"

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION = 0x00010400
# SPIR-V 1.4 cooperative-matrix opcodes (KHR).
OP_TYPE_COOPERATIVE_MATRIX = 4456
OP_COOPERATIVE_MATRIX_MUL_ADD = 4459

CANDIDATE = {
    "wmma_gemm_f16_f32acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.spv",
        "source_sha256": "239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc",
        "source_bytes": 13409,
        "binary_sha256": "e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65",
        "binary_bytes": 12720,
    },
    "wmma_gemm_f16_f16acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.spv",
        "source_sha256": "43a62481f1cf23d96505b39ec8226deb9f4a02c3f054d3c6a7646ba10649f05e",
        "source_bytes": 10246,
        "binary_sha256": "b040095e481f9de7625df7a2d5bc743027629468e2bb553b571ee0f1237ad73a",
        "binary_bytes": 12736,
    },
}

WMMA_RS = "hierarchos-vulkan/src/wmma.rs"
WMMA_BENCH = "hierarchos-vulkan/src/bin/wmma_bench.rs"
ABBA_RUNNER = "hierarchos-vulkan/validation/wmma_abba_run.sh"
ABBA_ANALYSER = "hierarchos-vulkan/validation/wmma_abba_analyze.py"

# The only implementation paths this change is allowed to touch. Every other
# path under these prefixes is the arithmetic, dispatch, fallback or tolerance
# surface, none of which the change may alter.
PRODUCTION_PATHS = (
    {spec["source"] for spec in CANDIDATE.values()}
    | {spec["binary"] for spec in CANDIDATE.values()}
    | {WMMA_RS, WMMA_BENCH}
)
PROTECTED_PREFIXES = (
    "hierarchos-vulkan/src/",
    "hierarchos-vulkan/shaders/",
    "hierarchos-inference/src/",
    "hierarchos-native-cli/src/",
    "hierarchos-gui/src/",
)
FORBIDDEN_ARTIFACT = re.compile(
    r"\.(safetensors|bin|pt|pth|gguf|onnx|exe|dll|so|dylib)$", re.IGNORECASE)
EXPECTED_FLAGS = ["-fshader-stage=compute", "--target-env=vulkan1.2",
                  "--target-spv=spv1.4", "-O"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def spirv_summary(data: bytes) -> dict:
    """Header and cooperative-matrix instruction counts of a SPIR-V module.

    Parsed directly so the structural assertions need no external toolchain. A
    module whose instruction stream does not tile the file exactly is rejected.
    """
    require(len(data) >= 20 and len(data) % 4 == 0,
            "SPIR-V module is not a whole number of 32-bit words")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    require(words[0] == SPIRV_MAGIC, "not a SPIR-V module (bad magic)")
    counts: dict[int, int] = {}
    index = 5
    while index < len(words):
        word = words[index]
        length = word >> 16
        require(length > 0, "zero-length SPIR-V instruction")
        require(index + length <= len(words),
                "SPIR-V instruction stream overruns the module")
        opcode = word & 0xFFFF
        counts[opcode] = counts.get(opcode, 0) + 1
        index += length
    require(index == len(words), "SPIR-V instruction stream does not tile the module")
    return {"version": words[1], "bound": words[3],
            "cooperative_mul_adds": counts.get(OP_COOPERATIVE_MATRIX_MUL_ADD, 0),
            "cooperative_types": counts.get(OP_TYPE_COOPERATIVE_MATRIX, 0)}


def check_spirv_structure(module: dict, stem: str) -> None:
    require(module["version"] == SPIRV_VERSION,
            f"{stem}: SPIR-V version is not 1.4")
    require(module["cooperative_types"] >= 1,
            f"{stem}: no cooperative matrix type in the module")
    require(module["cooperative_mul_adds"] == 4,
            f"{stem}: {module['cooperative_mul_adds']} cooperative multiply-add(s), not 4")


def check_artifacts(root: Path) -> dict:
    modules = {}
    for stem, spec in CANDIDATE.items():
        source, binary = root / spec["source"], root / spec["binary"]
        require(source.is_file(), f"missing shader source {spec['source']}")
        require(binary.is_file(), f"missing shader binary {spec['binary']}")
        require(source.stat().st_size == spec["source_bytes"],
                f"Candidate A source size changed: {spec['source']}")
        require(binary.stat().st_size == spec["binary_bytes"],
                f"Candidate A binary size changed: {spec['binary']}")
        require(digest(source) == spec["source_sha256"],
                f"Candidate A source changed: {spec['source']}")
        require(digest(binary) == spec["binary_sha256"],
                f"Candidate A binary changed: {spec['binary']}")
        modules[stem] = spirv_summary(binary.read_bytes())
    return modules


def check_source(root: Path, stem: str, spec: dict) -> None:
    text = read(root / spec["source"])
    require(text.count("f16vec2 a[]") == 1 and text.count("f16vec2 b[]") == 1,
            f"{stem}: operands are not declared as packed f16vec2 words")
    require(text.count("bool wide_a = (pc.lda & 1u) == 0u;") == 1
            and text.count("bool wide_b = (pc.ldb & 1u) == 0u;") == 1,
            f"{stem}: leading-dimension alignment gate changed")
    require(text.count("f16vec2 pair = a[off >> 1u];") == 2
            and text.count("f16vec2 pair = b[off >> 1u];") == 2,
            f"{stem}: packed operand loads changed")
    require(text.count("a[off >> 1u][off & 1u]") == 2
            and text.count("b[off >> 1u][off & 1u]") == 2,
            f"{stem}: masked 16-bit fallback changed")
    require(text.count("coopMatMulAdd(") == 4,
            f"{stem}: cooperative multiply-add count changed")
    require(text.count("coopMatLoad(") == 4,
            f"{stem}: fragment loads changed")


def check_policy(root: Path) -> None:
    text = read(root / WMMA_RS)
    require("pub const WMMA_MIN_PROFITABLE_MACS: u64 = 16_000_000;" in text,
            "profitability floor changed")
    require(">= WMMA_MIN_PROFITABLE_MACS" in text,
            "cooperative dispatch predicate changed")
    require("assert!(WMMA_MIN_PROFITABLE_MACS > 128 * 256 * 256);" in text,
            "profitability floor sanity assertion changed")


def check_boundaries(root: Path) -> None:
    text = read(root / WMMA_RS)
    for shape in ("(64, 48, 19)", "(19, 48, 64)"):
        require(shape in text, f"new boundary shape is missing: {shape}")
    for shape in ("(64usize, 48usize, 40usize)", "(33, 17, 19)", "(128, 128, 96)"):
        require(shape in text, f"existing boundary shape is missing: {shape}")


def check_benchmark(root: Path) -> None:
    bench = read(root / WMMA_BENCH)
    require("--dump-samples" in bench and "dump_samples" in bench,
            "benchmark sample export is missing")
    require("raw_lines" in bench and "samples_s" in bench,
            "raw per-iteration sample rows are missing")
    for path in (ABBA_RUNNER, ABBA_ANALYSER):
        require((root / path).is_file(), f"paired ABBA harness is incomplete: {path}")
    script = read(root / ABBA_RUNNER)
    require("--dump-samples" in script,
            "ABBA runner does not export raw samples")
    for default in ("PAIRS:-8", "ROUNDS:-4", "ITERS:-20"):
        require(default in script, f"ABBA runner default changed: {default}")
    require("index.tsv" in script, "ABBA runner does not index its runs")


def hunks(text: str) -> tuple[list[str], list[str]]:
    """Added and removed content lines of a unified diff, without the headers."""
    added, removed = [], []
    for line in text.splitlines():
        if line.startswith(("+++", "---", "@@", "diff --git", "index ", "new file ",
                            "deleted file ", "similarity index", "rename ")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def check_wmma_diff(text: str) -> None:
    """Only the boundary shape list and its comment may change in wmma.rs."""
    added, removed = hunks(text)
    require(len(removed) == 1,
            "wmma.rs changed outside the boundary shape list")
    require(removed[0].strip().startswith(
                "for (rows, columns, reduction) in [(64usize, 48usize, 40usize)"),
            "wmma.rs removed a line other than the boundary shape list")
    for line in added:
        stripped = line.strip()
        allowed = (stripped.startswith("//")
                   or stripped == "for (rows, columns, reduction) in ["
                   or stripped == "]"
                   or re.fullmatch(r"\(\d+usize, \d+usize, \d+usize\),", stripped)
                   or re.fullmatch(r"\(\d+, \d+, \d+\),", stripped))
        require(bool(allowed),
                f"wmma.rs changed a line outside the boundary shape list: {stripped!r}")


def check_benchmark_diff(text: str) -> None:
    """The benchmark's timing and statistics statements may not change."""
    added, removed = hunks(text)
    require(len(removed) == 1 and "--json" in removed[0] and "dump-samples" not in removed[0],
            "the benchmark help string is not the only removed line")
    for line in added:
        for banned in ("median(", "Instant::now", "Duration", "rounds =", "iterations ="):
            require(banned not in line,
                    f"benchmark methodology changed: {line.strip()!r}")
    require(any("--dump-samples" in line for line in added),
            "the sample-export flag was not added")
    require(any("dump_samples" in line for line in added),
            "the sample-export plumbing was not added")


def check_changed(changed) -> None:
    changed = set(changed)
    production = {path for path in changed if path.startswith(PROTECTED_PREFIXES)}
    unexpected = production ^ set(PRODUCTION_PATHS)
    require(not unexpected,
            "protected source changed outside the Candidate A paths: "
            + ", ".join(sorted(unexpected)))
    require(not any(Path(path).name in ("Cargo.toml", "Cargo.lock") for path in changed),
            "a dependency manifest changed in a WMMA staging diff")
    for path in sorted(changed):
        if FORBIDDEN_ARTIFACT.search(path) is None:
            continue
        require(path in PRODUCTION_PATHS,
                f"model weight or prebuilt binary added to the change: {path}")


def check_reproduction(receipt: dict, root: Path) -> None:
    require(receipt.get("schema") == 1
            and receipt.get("kind") == "wmma-candidate-a-spirv-reproduction",
            "unexpected reproduction receipt schema")
    require(receipt.get("pass") is True, "recorded reproduction did not pass")
    require(receipt.get("compiler") == "glslc",
            "recorded reproduction used a different compiler")
    require(list(receipt.get("flags") or []) == EXPECTED_FLAGS,
            "production shader compiler flags changed")
    require(receipt.get("spirv_val") == "vulkan1.2",
            "SPIR-V validation target changed")
    rows = receipt.get("shaders") or {}
    require(set(rows) == set(CANDIDATE),
            "reproduction receipt does not cover both modules")
    for stem, spec in CANDIDATE.items():
        row = rows[stem]
        require(row.get("source_sha256") == spec["source_sha256"],
                f"{stem}: recorded source differs from the qualified source")
        require(row.get("embedded_sha256") == spec["binary_sha256"],
                f"{stem}: recorded artifact differs from the qualified artifact")
        require(row.get("rebuilt_sha256") == row.get("embedded_sha256"),
                f"{stem}: the rebuilt SPIR-V does not reproduce the committed artifact")
        require(row.get("cooperative_mul_adds") == 4,
                f"{stem}: the reproduced module does not hold four multiply-adds")
        require(row.get("spirv_val") == "clean",
                f"{stem}: the recorded SPIR-V validation is not clean")
        require(digest(root / spec["source"]) == row["source_sha256"],
                f"{stem}: the source in this tree differs from the record")
        require(digest(root / spec["binary"]) == row["embedded_sha256"],
                f"{stem}: the artifact in this tree differs from the record")


def validate(root: Path, *, changed, diff, reproduction: dict,
             require_reproduction: bool = True) -> dict:
    """Every Candidate A assertion, without touching git or the network."""
    root = Path(root)
    modules = check_artifacts(root)
    for stem, spec in CANDIDATE.items():
        check_source(root, stem, spec)
        check_spirv_structure(modules[stem], stem)
    check_policy(root)
    check_boundaries(root)
    check_benchmark(root)
    check_changed(changed)
    require(WMMA_RS in diff, "the wmma.rs diff is missing")
    require(WMMA_BENCH in diff, "the benchmark diff is missing")
    check_wmma_diff(diff[WMMA_RS])
    check_benchmark_diff(diff[WMMA_BENCH])
    if require_reproduction:
        check_reproduction(reproduction, root)
    return modules


def reproduce(root: Path, receipt: dict) -> None:
    """Recompile both shaders and require the committed bytes."""
    compiler = shutil.which(receipt.get("compiler") or "glslc")
    require(compiler is not None,
            "glslc is not on PATH, so the shader binaries cannot be re-derived here")
    validator = shutil.which("spirv-val")
    with tempfile.TemporaryDirectory(prefix="candidate-a-") as scratch:
        for stem, spec in CANDIDATE.items():
            rebuilt = Path(scratch) / f"{stem}.spv"
            run = subprocess.run(
                [compiler, *receipt["flags"], "-o", str(rebuilt), str(root / spec["source"])],
                capture_output=True, text=True)
            require(run.returncode == 0,
                    f"{stem}: shader compilation failed: "
                    + (run.stdout + run.stderr).strip()[-300:])
            require(digest(rebuilt) == spec["binary_sha256"],
                    f"{stem}: recompiled SPIR-V does not match the committed artifact")
            if validator:
                check = subprocess.run(
                    [validator, "--target-env", receipt["spirv_val"],
                     str(root / spec["binary"])], capture_output=True, text=True)
                require(check.returncode == 0,
                        f"{stem}: spirv-val rejected the committed module: "
                        + check.stderr.strip()[-300:])


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    require(result.returncode == 0,
            f"git {' '.join(args)} failed: {result.stderr.strip()[:200]}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base", default=BASE_REVISION,
                        help="recorded base commit the change is diffed against")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--reproduce", action="store_true",
                        help="recompile both shaders and require byte identity")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        ancestor = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", args.base, "HEAD"],
            capture_output=True)
        require(ancestor.returncode == 0,
                f"recorded base {args.base[:12]} is not an ancestor of HEAD")
        changed = set(git("diff", "--name-only", f"{args.base}...HEAD", cwd=root).split())
        diff = {path: git("diff", f"{args.base}...HEAD", "--", path, cwd=root)
                for path in (WMMA_RS, WMMA_BENCH)}
        receipt = json.loads(args.fixture.read_text(encoding="utf-8"))
        validate(root, changed=changed, diff=diff, reproduction=receipt)
        if args.reproduce:
            reproduce(root, receipt)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"CANDIDATE A PR VALIDATION: FAIL ({error})")
        return 1
    mode = ("recompiled and byte-identical" if args.reproduce
            else "recorded reproduction receipt verified")
    print("CANDIDATE A PR VALIDATION: PASS "
          f"({len(changed)} path(s) changed against {args.base[:12]}, "
          "4 cooperative multiply-adds per module, 16,000,000-MAC floor unchanged, "
          f"{mode}; no GPU timing is claimed by hosted CI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Current-implementation gate for the Phase LXIV RDNA3 WMMA accuracy work.

Phase LXII is **history** and stays verifiable exactly as it was recorded: the
unmodified `check_candidate_a.py` still gates the Phase LXII tree at its recorded
revision (see `.github/workflows/candidate-a.yml`, which materialises that
revision and runs the original gate there). This gate asks the complementary
question about the tree under review: does *this* tree carry the Phase LXIV
implementation, with the Phase LXII performance module untouched, and the
recorded numerical result reproduced?

It verifies:

* `wmma_gemm_f16_f32acc.{comp,spv}` are still the exact Phase LXII bytes, so the
  measured performance path cannot drift while accuracy work lands;
* `wmma_gemm_f16_f16acc.{comp,spv}` are the exact Phase LXIV bytes, both modules
  are SPIR-V 1.4, and the cooperative multiply-add counts are 4 and 8;
* the split reduction is structurally present in the source: two accumulator
  banks, the alternating-slice selector, the FP16 merge, and no FP32 accumulator
  fragment anywhere in the module (so an FP32 accumulation cannot be reported as
  FP16 accumulation);
* the packed candidate A operand staging is still exactly as qualified;
* every numerical tolerance constant, the 16,000,000-MAC profitability floor and
  its dispatch predicate are unchanged;
* the matched-oracle surface still offers the historical `tile16` model and the
  two cadence-matched models, so no reference was removed or replaced;
* the recorded qualification receipt reproduces: 37 gates, the exact four
  remaining tile16-model/unqualified labels, zero FP32-budget failures, and a
  clean 37/37 under the contract-matched `split2` oracle;
* no dependency manifest, model weight or prebuilt binary was added, and no
  protected arithmetic path outside the named files changed.

It never claims a GPU measurement on a hosted runner: the numerical record is the
receipt measured on the RDNA3 adapter, and `--reproduce` re-derives the SPIR-V
byte-for-byte on a host with the pinned toolchain.
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
FIXTURE = HERE / "fixtures" / "wmma_lxiv_accuracy_receipt.json"

# The exact base the Phase LXII performance record and this change share.
BASE_REVISION = "81ac82fa972d84025129e7c3f0e58fdece7df786"
# The revision at which the Phase LXII candidate A artifacts were qualified.
PHASE_LXII_REVISION = "7c49558b8155a351b30893a6d8f318d4e063d1bd"

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION = 0x00010400
OP_TYPE_COOPERATIVE_MATRIX = 4456
OP_COOPERATIVE_MATRIX_MUL_ADD = 4459

CURRENT = {
    "wmma_gemm_f16_f32acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f32acc.spv",
        "source_sha256": "239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc",
        "source_bytes": 13409,
        "binary_sha256": "e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65",
        "binary_bytes": 12720,
        "cooperative_mul_adds": 4,
        "role": "phase-lxii-unchanged",
    },
    "wmma_gemm_f16_f16acc": {
        "source": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.comp",
        "binary": "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc.spv",
        "source_sha256": "7fc07d0c88c53b52a760c8785df7ae98c280a21af8c9da28abeb2f6d6a40495a",
        "source_bytes": 15062,
        "binary_sha256": "1a02ed98ff61c68ac4f30154e9b9797a5275c896b5743ab78fed78d137d4047f",
        "binary_bytes": 14976,
        "cooperative_mul_adds": 8,
        "role": "phase-lxiv-split-reduction",
    },
}

WMMA_RS = "hierarchos-vulkan/src/wmma.rs"
WMMA_QUALIFY = "hierarchos-vulkan/src/bin/wmma_qualify.rs"
# Phase LXII's paired-benchmark harness. The cumulative diff against the recorded
# base contains it, and Phase LXIV does not touch it.
WMMA_BENCH = "hierarchos-vulkan/src/bin/wmma_bench.rs"

PRODUCTION_PATHS = (
    {spec["source"] for spec in CURRENT.values()}
    | {spec["binary"] for spec in CURRENT.values()}
    | {WMMA_RS, WMMA_QUALIFY, WMMA_BENCH}
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

# Exact text of every numerical limit. A relaxation, a precision substitution or
# a reworded constant is rejected rather than interpreted.
TOLERANCES = (
    "pub const MIXED_VS_FP32_MAX_ABS: f64 = 2.0e-2;",
    "pub const MIXED_VS_FP32_RMS: f64 = 3.0e-3;",
    "pub const MIXED_VS_MATCHED_ORACLE_MAX_ABS: f64 = 1.0e-3;",
    "pub const FULL_VS_FP32_MAX_ABS: f64 = 5.0e-2;",
    "pub const FULL_VS_FP32_RMS: f64 = 1.0e-2;",
    "pub const FULL_VS_MATCHED_ORACLE_MAX_ABS: f64 = 4.0e-3;",
)

# The four gates that the historical single-accumulator module also failed, and
# that the historical `tile16` model cannot qualify: it assumes one rounding per
# 16-wide K slice, while the hardware rounds on the dot2 cadence. They are named
# exactly so a different four can never be reported as "the same four".
EXPECTED_TILE16_FAILURES = (
    "128x128x96 wmma-full-fp16 vs matched oracle",
    "128x128x96 trans_a=true trans_b=false vs oracle",
    "128x128x96 trans_a=false trans_b=true vs oracle",
    "128x128x96 trans_a=true trans_b=true vs oracle",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def spirv_summary(data: bytes) -> dict:
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


def check_artifacts(root: Path) -> dict:
    modules = {}
    for stem, spec in CURRENT.items():
        source, binary = root / spec["source"], root / spec["binary"]
        require(source.is_file(), f"missing shader source {spec['source']}")
        require(binary.is_file(), f"missing shader binary {spec['binary']}")
        require(source.stat().st_size == spec["source_bytes"],
                f"{stem}: source size changed")
        require(binary.stat().st_size == spec["binary_bytes"],
                f"{stem}: binary size changed")
        require(digest(source) == spec["source_sha256"],
                f"{stem}: source differs from the qualified bytes")
        require(digest(binary) == spec["binary_sha256"],
                f"{stem}: artifact differs from the qualified bytes")
        modules[stem] = spirv_summary(binary.read_bytes())
        require(modules[stem]["version"] == SPIRV_VERSION,
                f"{stem}: SPIR-V version is not 1.4")
        require(modules[stem]["cooperative_types"] >= 1,
                f"{stem}: no cooperative matrix type in the module")
        require(modules[stem]["cooperative_mul_adds"] == spec["cooperative_mul_adds"],
                f"{stem}: {modules[stem]['cooperative_mul_adds']} cooperative "
                f"multiply-add(s), not {spec['cooperative_mul_adds']}")
    return modules


def check_split_source(root: Path) -> None:
    text = read(root / CURRENT["wmma_gemm_f16_f16acc"]["source"])
    # Two banks, selected by the K-slice parity.
    require(text.count("alt00") >= 2 and text.count("alt11") >= 2,
            "the second accumulator bank is missing")
    require(text.count("if ((slice & 1u) == 0u)") == 1,
            "the alternating-slice selector changed")
    require(text.count("slice++;") == 1,
            "the K-slice counter changed")
    # The merge must be an FP16 addition, not a widening one.
    require(text.count("float16_t merged = float16_t(c_part[e] + float(c_tile[e]));") == 4,
            "the FP16 bank merge changed")
    require(text.count("gl_MatrixUseAccumulator>(float16_t(0.0))") == 8,
            "the accumulator bank initialisation changed")
    # No FP32 accumulator fragment: this arm must never silently accumulate in
    # FP32 and still be reported as FP16 accumulation.
    require("coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>"
            not in text,
            "an FP32 accumulator fragment was introduced into the FP16 arm")
    # Candidate A packed staging must be untouched.
    require(text.count("f16vec2 a[]") == 1 and text.count("f16vec2 b[]") == 1,
            "operands are not declared as packed f16vec2 words")
    require(text.count("bool wide_a = (pc.lda & 1u) == 0u;") == 1
            and text.count("bool wide_b = (pc.ldb & 1u) == 0u;") == 1,
            "leading-dimension alignment gate changed")
    require(text.count("f16vec2 pair = a[off >> 1u];") == 2
            and text.count("f16vec2 pair = b[off >> 1u];") == 2,
            "packed operand loads changed")
    require(text.count("a[off >> 1u][off & 1u]") == 2
            and text.count("b[off >> 1u][off & 1u]") == 2,
            "masked 16-bit fallback changed")
    require(text.count("coopMatLoad(") == 4, "fragment loads changed")


def check_policy(root: Path) -> None:
    text = read(root / WMMA_RS)
    for constant in TOLERANCES:
        require(constant in text, f"numerical limit changed or missing: {constant}")
    require("pub const WMMA_MIN_PROFITABLE_MACS: u64 = 16_000_000;" in text,
            "profitability floor changed")
    require(">= WMMA_MIN_PROFITABLE_MACS" in text,
            "cooperative dispatch predicate changed")
    require("assert!(WMMA_MIN_PROFITABLE_MACS > 128 * 256 * 256);" in text,
            "profitability floor sanity assertion changed")
    # The matched-oracle surface must keep the historical model and add the
    # cadence-matched one; a removed reference cannot be reported as a pass.
    for variant in ("Fp32,", "Fp16TileK,", "Fp16Dot2,", "Fp16Split2,"):
        require(re.search(rf"^\s*{re.escape(variant)}\s*$", text, re.MULTILINE)
                is not None,
                f"matched-oracle variant removed: {variant.rstrip(',')}")
    qualify = read(root / WMMA_QUALIFY)
    for option in ('Some("tile16")', 'Some("rdna3-dot2")', 'Some("split2")'):
        require(option in qualify, f"oracle option removed: {option}")
    require("wmma_full_fp16_meets_fp32_budget_on_long_reductions" in text,
            "the long-reduction FP32 budget test is missing")


def check_changed(changed) -> None:
    changed = set(changed)
    production = {path for path in changed if path.startswith(PROTECTED_PREFIXES)}
    unexpected = production ^ set(PRODUCTION_PATHS)
    require(not unexpected,
            "protected source changed outside the qualified WMMA paths: "
            + ", ".join(sorted(unexpected)))
    require(not any(Path(path).name in ("Cargo.toml", "Cargo.lock") for path in changed),
            "a dependency manifest changed in an arithmetic change")
    for path in sorted(changed):
        if FORBIDDEN_ARTIFACT.search(path) is None:
            continue
        require(path in PRODUCTION_PATHS,
                f"model weight or prebuilt binary added to the change: {path}")


def gate_rows(rows) -> dict:
    return {row["label"]: row for row in rows}


def check_receipt(receipt: dict) -> None:
    require(receipt.get("schema") == 1
            and receipt.get("kind") == "wmma-lxiv-accuracy-receipt",
            "unexpected accuracy receipt schema")
    require(receipt.get("adapter") == "AMD Radeon Graphics 1002:15bf",
            "the receipt is not from the qualified RDNA3 adapter")
    require(list(receipt.get("flags") or []) == EXPECTED_FLAGS,
            "production shader compiler flags changed")
    require(receipt.get("spirv_val") == "vulkan1.2",
            "SPIR-V validation target changed")
    shaders = receipt.get("shaders") or {}
    for stem, spec in CURRENT.items():
        row = shaders.get(stem)
        require(row is not None, f"receipt does not cover {stem}")
        require(row.get("source_sha256") == spec["source_sha256"],
                f"{stem}: recorded source differs from the qualified source")
        require(row.get("binary_sha256") == spec["binary_sha256"],
                f"{stem}: recorded artifact differs from the qualified artifact")
        require(row.get("rebuilt_sha256") == spec["binary_sha256"],
                f"{stem}: the rebuilt SPIR-V does not reproduce the artifact")
        require(row.get("cooperative_mul_adds") == spec["cooperative_mul_adds"],
                f"{stem}: recorded multiply-add count changed")
        require(row.get("spirv_val") == "clean",
                f"{stem}: the recorded SPIR-V validation is not clean")

    runs = receipt.get("runs") or {}
    for mode in ("mixed", "full-fp16-tile16", "full-fp16-split2"):
        require(mode in runs, f"qualification run missing from the receipt: {mode}")
    mixed = runs["mixed"]
    require(mixed["gates"] == 37 and mixed["failed"] == 0,
            "the mixed qualification is no longer 37/37")
    split2 = runs["full-fp16-split2"]
    require(split2["gates"] == 37 and split2["failed"] == 0,
            "the contract-matched full-FP16 qualification is not 37/37")
    tile16 = runs["full-fp16-tile16"]
    require(tile16["gates"] == 37,
            "the historical full-FP16 qualification no longer runs 37 gates")
    require(tile16["failed"] == len(EXPECTED_TILE16_FAILURES),
            "the historical-oracle failure count changed")
    failed = sorted(row["label"] for row in tile16["results"] if not row["passed"])
    require(failed == sorted(EXPECTED_TILE16_FAILURES),
            "a different set of historical-oracle gates fails: " + "; ".join(failed))

    # Accuracy is graded against the unchanged absolute limits, so a gate that
    # compares against the FP32 reference may never be recorded as failing.
    for name, run in runs.items():
        for row in run["results"]:
            if "vs FP32" not in row["label"]:
                continue
            require(row["passed"],
                    f"{name}: FP32-budget gate recorded as failing: {row['label']}")
            require(row["max_abs"] <= row["max_abs_limit"],
                    f"{name}: {row['label']} exceeds its recorded limit")
            if row.get("rms_limit") is not None:
                require(row["rms"] <= row["rms_limit"],
                        f"{name}: {row['label']} exceeds its recorded RMS limit")

    # The three geometries that previously failed the FP32 budget must be
    # recorded as passing, with their measured values retained.
    previously = receipt.get("previously_failing") or {}
    require(set(previously) == {
        "128x128x96 trans_a=true trans_b=false vs FP32",
        "128x128x96 trans_a=false trans_b=true vs FP32",
        "128x128x96 trans_a=true trans_b=true vs FP32",
    }, "the previously failing FP32 gates are not all recorded")
    for label, record in previously.items():
        require(record["before"] > record["limit"],
                f"{label}: the recorded baseline was not actually failing")
        require(record["after"] <= record["limit"],
                f"{label}: the recorded result does not pass the unchanged limit")
        require(record["limit"] == 5.0e-2,
                f"{label}: the recorded limit is not the unchanged 0.05")
    require(receipt.get("pass") is True, "recorded accuracy result did not pass")


def validate(root: Path, *, changed, receipt: dict) -> dict:
    root = Path(root)
    modules = check_artifacts(root)
    check_split_source(root)
    check_policy(root)
    check_changed(changed)
    check_receipt(receipt)
    return modules


def reproduce(root: Path, receipt: dict) -> None:
    compiler = shutil.which(receipt.get("compiler") or "glslc")
    require(compiler is not None,
            "glslc is not on PATH, so the shader binaries cannot be re-derived here")
    validator = shutil.which("spirv-val")
    with tempfile.TemporaryDirectory(prefix="phase-lxiv-") as scratch:
        for stem, spec in CURRENT.items():
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
        for revision in (args.base, PHASE_LXII_REVISION):
            ancestor = subprocess.run(
                ["git", "-C", str(root), "merge-base", "--is-ancestor", revision, "HEAD"],
                capture_output=True)
            require(ancestor.returncode == 0,
                    f"recorded revision {revision[:12]} is not an ancestor of HEAD")
        changed = set(git("diff", "--name-only", f"{args.base}...HEAD", cwd=root).split())
        receipt = json.loads(args.fixture.read_text(encoding="utf-8"))
        validate(root, changed=changed, receipt=receipt)
        if args.reproduce:
            reproduce(root, receipt)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"PHASE LXIV CURRENT-CANDIDATE VALIDATION: FAIL ({error})")
        return 1
    mode = ("recompiled and byte-identical" if args.reproduce
            else "recorded reproduction receipt verified")
    print("PHASE LXIV CURRENT-CANDIDATE VALIDATION: PASS "
          f"({len(changed)} path(s) changed against {args.base[:12]}, "
          "Phase LXII f32acc artifacts byte-identical, 8 cooperative multiply-adds "
          "in the split module, 16,000,000-MAC floor and every tolerance unchanged, "
          f"{mode}; no GPU timing is claimed by hosted CI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the Phase LXXXVII investigation modules from the qualified sources.

Every module here is derived from a byte-pinned source by exact textual edit:
the committed Phase LXXIX P3 module (`4e448e3c…`) or the recovered Phase LXXXVI
lean module (`a75db2d3…`). If a fragment is not found exactly once the generator
raises instead of producing a module that measures something else.

Instruments produced (all are *additions* to the existing set; none replaces a
committed production module):

  p3_batch      P3 with an unrolled, bounds-check-free staging fast path taken
                when the staged 32x16 slab is fully in bounds and the operand
                is pair-aligned. Same values, same slots, same barriers.
                **Bit-identical arithmetic** — candidate C1 (staging latency).
  p3_leanpad    lean with `bank_sum` widened from [4][256] to [4][448], i.e.
                9,216 B of LDS instead of 6,144 B with *zero* extra
                instructions (only the row stride changes). This is the
                LDS-matched control for the tall-K investigation: if lean's
                shape-dependent behaviour is a residency effect, leanpad must
                behave like P3; if it is an epilogue-code effect, leanpad must
                behave like lean.
  p3_pad        P3 with `bank_sum` widened to [4][576] (14,336 B of LDS) — a
                second point on the same curve, to test whether the LDS
                footprint changes runtime at all on these shapes.
  ablnoepi      The Phase LXXXVI `abl_staging_only`-family instrument that was
                never measured: P3 with the per-pair LDS round trip, the float32
                bank sum and its barriers removed. WRONG NUMERICALLY, profiling
                instrument only. Regenerated here from the preserved
                `ablations/abl_no_epilogue.comp` so the byte identity to
                `93168bd6…` is re-derived rather than assumed.

Usage: make_candidates.py <out-dir>
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VULKAN = HERE.parents[1]
REPO = VULKAN.parent
P3 = VULKAN / "shaders" / "wmma_gemm_f16_f16acc_twowave_p3.comp"
LEAN = (
    VULKAN
    / "validation"
    / "phase_lxxxvi_wmma_optimization"
    / "candidates"
    / "wmma_gemm_f16_f16acc_twowave_p3_lean.comp"
)
ABL_NO_EPILOGUE = (
    VULKAN
    / "validation"
    / "phase_lxxxvi_wmma_optimization"
    / "ablations"
    / "abl_no_epilogue.comp"
)

STAGING_LOOP = "for (uint p = tid; p < 256u; p += threads) {"

# Indentation of the staging loops inside the P3 source.
IND = " " * 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_brace(text: str, opened: int) -> int:
    depth = 0
    index = opened
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise SystemExit("unbalanced braces")


def require_once(text: str, fragment: str, label: str) -> int:
    count = text.count(fragment)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.index(fragment)


# --- batched staging fast paths ------------------------------------------------
#
# `layout(local_size_x = 64)` makes `threads == 64` a compile-time fact, so the
# four staging iterations per thread are `p = tid, tid+64, tid+128, tid+192`,
# i.e. tile 0 with e in {tid, tid+64} and tile 1 with the same two e values.
# The fast path issues all four 32-bit operand loads before any shared store and
# drops the per-element bounds tests, which the uniform predicate makes provably
# unnecessary. The shared stores write the same values to the same slots as the
# masked loop; the arithmetic contract is untouched.

A_ROW_MAJOR = """\
uint ea = tid;
uint eb = tid + 64u;
uint ra = ea >> 3u;
uint ca = (ea & 7u) << 1u;
uint rb = eb >> 3u;
uint cb = (eb & 7u) << 1u;
uint gkf0 = k0 + ca;
uint gkf1 = k0 + cb;
f16vec2 p0 = a[((m0 + ra) * pc.lda + gkf0) >> 1u];
f16vec2 p1 = a[((m0 + rb) * pc.lda + gkf1) >> 1u];
f16vec2 p2 = a[((m0 + 16u + ra) * pc.lda + gkf0) >> 1u];
f16vec2 p3 = a[((m0 + 16u + rb) * pc.lda + gkf1) >> 1u];
a_tile[0][ra * 16u + ca] = p0.x;
a_tile[0][ra * 16u + ca + 1u] = p0.y;
a_tile[0][rb * 16u + cb] = p1.x;
a_tile[0][rb * 16u + cb + 1u] = p1.y;
a_tile[1][ra * 16u + ca] = p2.x;
a_tile[1][ra * 16u + ca + 1u] = p2.y;
a_tile[1][rb * 16u + cb] = p3.x;
a_tile[1][rb * 16u + cb + 1u] = p3.y;"""

A_TRANSPOSED = """\
uint ea = tid;
uint eb = tid + 64u;
uint ra = (ea >> 4u) << 1u;
uint ca = ea & 15u;
uint rb = (eb >> 4u) << 1u;
uint cb = eb & 15u;
uint gkf0 = k0 + ca;
uint gkf1 = k0 + cb;
f16vec2 p0 = a[(gkf0 * pc.lda + m0 + ra) >> 1u];
f16vec2 p1 = a[(gkf1 * pc.lda + m0 + rb) >> 1u];
f16vec2 p2 = a[(gkf0 * pc.lda + m0 + 16u + ra) >> 1u];
f16vec2 p3 = a[(gkf1 * pc.lda + m0 + 16u + rb) >> 1u];
a_tile[0][ra * 16u + ca] = p0.x;
a_tile[0][ra * 16u + ca + 16u] = p0.y;
a_tile[0][rb * 16u + cb] = p1.x;
a_tile[0][rb * 16u + cb + 16u] = p1.y;
a_tile[1][ra * 16u + ca] = p2.x;
a_tile[1][ra * 16u + ca + 16u] = p2.y;
a_tile[1][rb * 16u + cb] = p3.x;
a_tile[1][rb * 16u + cb + 16u] = p3.y;"""

B_ROW_MAJOR = """\
uint ea = tid;
uint eb = tid + 64u;
uint ra = ea >> 3u;
uint ca = (ea & 7u) << 1u;
uint rb = eb >> 3u;
uint cb = (eb & 7u) << 1u;
uint gkf0 = k0 + ra;
uint gkf1 = k0 + rb;
f16vec2 p0 = b[(gkf0 * pc.ldb + n0 + ca) >> 1u];
f16vec2 p1 = b[(gkf1 * pc.ldb + n0 + cb) >> 1u];
f16vec2 p2 = b[(gkf0 * pc.ldb + n0 + 16u + ca) >> 1u];
f16vec2 p3 = b[(gkf1 * pc.ldb + n0 + 16u + cb) >> 1u];
b_tile[0][ra * 16u + ca] = p0.x;
b_tile[0][ra * 16u + ca + 1u] = p0.y;
b_tile[0][rb * 16u + cb] = p1.x;
b_tile[0][rb * 16u + cb + 1u] = p1.y;
b_tile[1][ra * 16u + ca] = p2.x;
b_tile[1][ra * 16u + ca + 1u] = p2.y;
b_tile[1][rb * 16u + cb] = p3.x;
b_tile[1][rb * 16u + cb + 1u] = p3.y;"""

# B transposed [n][k]: the rank pair is along the reduction axis, so the k
# offset comes from the LDS row index and the n offset from the LDS column
# index. In the reference loop that is `gk = k0 + row`, `gn = n0 + tile*16 + col`
# with `row = (e>>4)<<1` and `col = e & 15`.
B_TRANSPOSED = """\
uint ea = tid;
uint eb = tid + 64u;
uint ra = (ea >> 4u) << 1u;
uint ca = ea & 15u;
uint rb = (eb >> 4u) << 1u;
uint cb = eb & 15u;
uint gkf0 = k0 + ra;
uint gkf1 = k0 + rb;
f16vec2 p0 = b[((n0 + ca) * pc.ldb + gkf0) >> 1u];
f16vec2 p1 = b[((n0 + cb) * pc.ldb + gkf1) >> 1u];
f16vec2 p2 = b[((n0 + 16u + ca) * pc.ldb + gkf0) >> 1u];
f16vec2 p3 = b[((n0 + 16u + cb) * pc.ldb + gkf1) >> 1u];
b_tile[0][ra * 16u + ca] = p0.x;
b_tile[0][ra * 16u + ca + 16u] = p0.y;
b_tile[0][rb * 16u + cb] = p1.x;
b_tile[0][rb * 16u + cb + 16u] = p1.y;
b_tile[1][ra * 16u + ca] = p2.x;
b_tile[1][ra * 16u + ca + 16u] = p2.y;
b_tile[1][rb * 16u + cb] = p3.x;
b_tile[1][rb * 16u + cb + 16u] = p3.y;"""

FAST_PATHS = [
    ("wide_a && m0 + 32u <= pc.m && k0 + 16u <= pc.k", A_ROW_MAJOR),
    ("wide_a && k0 + 16u <= pc.k && m0 + 32u <= pc.m", A_TRANSPOSED),
    ("wide_b && k0 + 16u <= pc.k && n0 + 32u <= pc.n", B_ROW_MAJOR),
    ("wide_b && n0 + 32u <= pc.n && k0 + 16u <= pc.k", B_TRANSPOSED),
]


def batch_source(text: str) -> str:
    starts = []
    cursor = 0
    while True:
        found = text.find(STAGING_LOOP, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if len(starts) != 4:
        raise SystemExit(f"expected 4 staging loops, found {len(starts)}")

    for index in reversed(range(4)):
        start = starts[index]
        if text.rfind("\n", 0, start) + 1 + 24 != start:
            raise SystemExit(f"staging loop {index} is not at the expected indent")
        opened = text.index("{", start)
        end = close_brace(text, opened)
        loop = text[start : end + 1]
        condition, body = FAST_PATHS[index]
        indented = "\n".join(
            (IND + "    " + line) if line else line for line in body.splitlines()
        )
        reference = "\n".join(IND + "    " + line for line in loop.splitlines())
        replacement = (
            f"if ({condition}) {{\n"
            f"{indented}\n"
            f"{IND}}} else {{\n"
            f"{reference}\n"
            f"{IND}}}"
        )
        text = text[:start] + replacement + text[end + 1 :]
    return text


def widen_bank_sum(text: str, columns: int) -> str:
    require_once(text, "shared float bank_sum[4][256];", "bank_sum declaration")
    return text.replace(
        "shared float bank_sum[4][256];",
        f"// LDS-footprint control (Phase LXXXVII): {columns} columns x 4 tiles x 4 B.\n"
        f"shared float bank_sum[4][{columns}];",
    )


def compile_spv(source: Path, output: Path) -> None:
    command = [
        "glslc",
        "-fshader-stage=compute",
        "--target-env=vulkan1.2",
        str(source),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"glslc failed for {source.name}:\n{completed.stdout}\n{completed.stderr}"
        )
    validated = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.2", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if validated.returncode != 0:
        raise SystemExit(
            f"spirv-val failed for {output.name}:\n{validated.stdout}{validated.stderr}"
        )


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    # The recovered Phase LXXXVI lean source is not valid UTF-8: it carries a
    # cp1252 em-dash in a comment. latin-1 round-trips every byte, so the derived
    # module keeps the original bytes and the SPIR-V identity stays meaningful.
    p3_text = P3.read_text(encoding="latin-1")
    lean_text = LEAN.read_text(encoding="latin-1")

    modules = {
        "wmma_gemm_f16_f16acc_twowave_p3_batch": batch_source(p3_text),
        "wmma_gemm_f16_f16acc_twowave_p3_leanpad": widen_bank_sum(lean_text, 448),
        "wmma_gemm_f16_f16acc_twowave_p3_pad": widen_bank_sum(p3_text, 576),
        "wmma_gemm_f16_f16acc_twowave_p3_ablnoepi": ABL_NO_EPILOGUE.read_text(
            encoding="latin-1"
        ),
    }

    for name, text in modules.items():
        source = out / f"{name}.comp"
        source.write_text(text, encoding="latin-1", newline="")
        spv = out / f"{name}.spv"
        compile_spv(source, spv)
        print(
            f"{name}: source {sha256(source)} ({source.stat().st_size} B) -> "
            f"spir-v {sha256(spv)} ({spv.stat().st_size} B)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

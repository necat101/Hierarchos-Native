#!/usr/bin/env python3
"""Generate the Phase LXXXVI ablation modules from the qualified P3 source.

These are **profiling instruments, not candidates**. Each one deletes a different
part of the P3 kernel so that a GPU-timestamp measurement can attribute the
production kernel's time to a mechanism instead of to a guess. None of them is
numerically valid and none may be dispatched as production; they exist only to
answer "where do the 5.31 ms go?" on the qualified adapter.

Every rewrite is an exact textual match against the committed P3 source. If the
source does not contain the fragment, the generator raises instead of silently
producing a module that measures something else.

Outputs (written to --out):
  abl_staging_only.comp / .spv   staging, address arithmetic, LDS stores and both
                                 per-slice barriers, but no cooperative-matrix
                                 load, no multiply-add and no bank-sum epilogue.
  abl_no_global.comp / .spv      the full kernel with the four global operand
                                 reads replaced by values derived from the same
                                 address, so every index computation and LDS
                                 store survives but no global traffic occurs.
  abl_no_epilogue.comp / .spv    the full kernel with the P3 bank-sum epilogue
                                 removed; the last pair's accumulators are stored
                                 through a single cooperative store.

Usage: make_ablations.py <p3-source> <out-dir>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

P3_RELATIVE = "hierarchos-vulkan/shaders/wmma_gemm_f16_f16acc_twowave_p3.comp"

MMA_BLOCK = """                    coopMatLoad(a_frag, a_tile[wave], 0, 16,
                                gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(b_frag0, b_tile[0], 0, 16,
                                gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(b_frag1, b_tile[1], 0, 16,
                                gl_CooperativeMatrixLayoutRowMajor);

                    if (sub == 0u) {
                        e0 = coopMatMulAdd(a_frag, b_frag0, e0);
                        e1 = coopMatMulAdd(a_frag, b_frag1, e1);
                    } else {
                        o0 = coopMatMulAdd(a_frag, b_frag0, o0);
                        o1 = coopMatMulAdd(a_frag, b_frag1, o1);
                    }
"""

EPILOGUE_BLOCK = """        // P3: add this pair's two banks to the float32 running sum, once, in
        // ascending bank order. No pair fold and no merge-tree level happens.
        ACCUM_PAIR(wave, tile0, e0, o0, pair);
        ACCUM_PAIR(wave, tile1, e1, o1, pair);
    }

    // The single rounding point, and the store, unchanged in masking and layout.
    STORE_SUM(tile0, row_base, n0);
    STORE_SUM(tile1, row_base, n0 + 16u);
}
"""

STAGING_ONLY_EPILOGUE = """    }

    // ABLATION (profiling instrument, not a candidate): no cooperative-matrix
    // load, no multiply-add and no bank-sum epilogue. The staged tiles are
    // consumed directly so the staging stores cannot be removed as dead code,
    // and both output tiles keep the production store's masking and geometry.
    for (uint e = lane; e < 256u; e += lanes) {
        uint gm = row_base + (e >> 4u);
        uint gn = n0 + (e & 15u);
        if (gm < pc.m && gn < pc.n) {
            c[gm * pc.ldc + gn] = float(a_tile[wave][e]) + float(b_tile[0][e]);
        }
    }
    for (uint e = lane; e < 256u; e += lanes) {
        uint gm = row_base + (e >> 4u);
        uint gn = n0 + 16u + (e & 15u);
        if (gm < pc.m && gn < pc.n) {
            c[gm * pc.ldc + gn] = float(a_tile[wave][e]) + float(b_tile[1][e]);
        }
    }
}
"""

NO_EPILOGUE_BLOCK = """unused"""


NO_EPILOGUE_INNER = """        // ABLATION (profiling instrument, not a candidate): keep the cooperative
        // stores but drop the per-pair barriers, the scalar LDS read-back, the
        // float32 additions and the bank_sum traffic.
        coopMatStore(e0, bank_tile[tile0], 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(o0, c_tile[wave], 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(e1, bank_tile[tile1], 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatStore(o1, c_tile[wave], 0, 16, gl_CooperativeMatrixLayoutRowMajor);
"""

NO_EPILOGUE_STORE = """    // ABLATION: store the last pair's cooperative staging directly.
    barrier();
    for (uint e = lane; e < 256u; e += lanes) {
        uint gm = row_base + (e >> 4u);
        uint gn = n0 + (e & 15u);
        if (gm < pc.m && gn < pc.n) {
            c[gm * pc.ldc + gn] = float(bank_tile[tile0][e]);
        }
    }
    for (uint e = lane; e < 256u; e += lanes) {
        uint gm = row_base + (e >> 4u);
        uint gn = n0 + 16u + (e & 15u);
        if (gm < pc.m && gn < pc.n) {
            c[gm * pc.ldc + gn] = float(bank_tile[tile1][e]);
        }
    }
}
"""

ACCUM_CALLS = """        ACCUM_PAIR(wave, tile0, e0, o0, pair);
        ACCUM_PAIR(wave, tile1, e1, o1, pair);
"""

STORE_CALLS = """    STORE_SUM(tile0, row_base, n0);
    STORE_SUM(tile1, row_base, n0 + 16u);
}
"""


def require_once(text: str, fragment: str, label: str) -> int:
    count = text.count(fragment)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.index(fragment)


def replace_once(text: str, fragment: str, replacement: str, label: str) -> str:
    require_once(text, fragment, label)
    return text.replace(fragment, replacement)


def close_brace(text: str, opened: int) -> int:
    """Index of the `}` that closes the `{` at ``opened``."""
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
    raise SystemExit("unbalanced braces while locating a global-load block")


def global_load_blocks(text: str) -> list[tuple[int, int]]:
    """Character spans of the four `if (wide_*) { ... } else { ... }` reads.

    Located by the guard, then closed by brace matching from the `{` that follows
    it, so a source edit that changes the body cannot silently leave the
    generator rewriting the wrong region.
    """
    spans = []
    for guard in ("if (wide_a &&", "if (wide_b &&"):
        start = 0
        while True:
            found = text.find(guard, start)
            if found < 0:
                break
            opened = text.index("{", found)
            index = close_brace(text, opened)
            # The scalar tail is an `else { ... }`; consume it as well, or the
            # rewrite leaves a dangling else.
            tail = index + 1
            while tail < len(text) and text[tail] in " \t\r\n":
                tail += 1
            if text.startswith("else", tail):
                else_open = text.index("{", tail)
                index = close_brace(text, else_open)
            spans.append((found, index + 1))
            start = index + 1
    if len(spans) != 4:
        raise SystemExit(f"expected 4 global-load blocks, found {len(spans)}")
    return sorted(spans)


def strip_indent(block: str, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(
        line[len(prefix):] if line.startswith(prefix) else line
        for line in block.splitlines()
    )


def abl_staging_only(text: str) -> str:
    text = replace_once(text, MMA_BLOCK, "", "staging-only: MMA block")
    text = replace_once(text, EPILOGUE_BLOCK, STAGING_ONLY_EPILOGUE, "staging-only: epilogue")
    return text


def abl_no_global(text: str) -> str:
    spans = global_load_blocks(text)
    for start, end in reversed(spans):
        line_start = text.rfind("\n", 0, start) + 1
        indent = start - line_start
        inner = strip_indent(text[start:end], indent + 4)
        replacement = (
            "// ABLATION (profiling instrument, not a candidate): the same address\n"
            f"{' ' * indent}// arithmetic, no global read. `off` still feeds the value.\n"
            f"{' ' * indent}v0 = float16_t(float(off & 255u) * 0.00390625);\n"
            f"{' ' * indent}v1 = float16_t(float((off >> 3u) & 63u) * 0.015625);"
        )
        del inner
        text = text[:start] + replacement + text[end:]
    return text


def abl_no_epilogue(text: str) -> str:
    text = replace_once(text, ACCUM_CALLS, NO_EPILOGUE_INNER, "no-epilogue: pair epilogue")
    text = replace_once(text, STORE_CALLS, NO_EPILOGUE_STORE, "no-epilogue: store")
    return text


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


def main() -> int:
    source = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = source.read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)

    ablations = {
        "abl_staging_only": abl_staging_only,
        "abl_no_global": abl_no_global,
        "abl_no_epilogue": abl_no_epilogue,
    }
    for name, rewrite in ablations.items():
        rewritten = rewrite(text)
        if rewritten == text:
            raise SystemExit(f"{name}: rewrite changed nothing")
        path = out / f"{name}.comp"
        path.write_text(rewritten, encoding="utf-8")
        compile_spv(path, out / f"{name}.spv")
        print(f"{name}: {path.stat().st_size} B source -> "
              f"{(out / (name + '.spv')).stat().st_size} B SPIR-V")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

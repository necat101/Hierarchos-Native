#!/usr/bin/env python3
"""Current-release gate for the v0.2.6 C1 staging promotion.

The v0.2.5 gate (`check_p3_release.py`) is **not** relaxed here: this gate runs
it in full, unmodified, as its first step. Everything that gate pins — the P3
source and SPIR-V bytes, `FullFp16Variant::P3` dispatching that module, the
four-variant inventory, the opt-in `off` default, the 16,000,000-MAC predicate
at every production admission point, the untouched Mixed route and the
selectable split8/two-bank/two-wave fallbacks — must still hold, and holding it
is a precondition of this gate's own result.

On top of that it pins what v0.2.6 adds:

* the C1 staging source and SPIR-V are the exact bytes recovered in
  Phase LXXXVII (`9857a8ad…` / `b0722420…`, 36,912 B) and are embedded from the
  committed artifact;
* `FullFp16Variant::P3` is the only variant that can reach C1, through
  `staged_spirv()`/`staged_shader_name()`; the other three return `None`;
* the production selection is the geometry contract, not a flag: `record_gemm`
  consults `batched_staging_eligible` only inside the `FullFp16Accumulate` arm
  and keeps the original module on every path that does not prove it;
* the predicate's body still contains the tile-coverage, packed-pair parity and
  32-bit-bounds conditions, and the capacity checks against both bound
  operands — so a weakened predicate fails here;
* per-dispatch staging counters exist and are only moved by Full FP16
  dispatches;
* no environment variable, CLI flag or variant name can force C1. The only
  variant knob remains `HIERARCHOS_WMMA_FULL_FP16_VARIANT`, and it cannot name
  a staging implementation;
* the pinned C1 source keeps the P3 barrier count, so the staging fast path
  adds no barrier inside a divergent branch.

CPU-side only: this gate never claims to have executed an adapter. The GPU
evidence is the recorded Phase LXXXVIII receipts in
`validation/phase_lxxxviii_c1_promotion/`.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import check_p3_release as p3

ROOT = p3.ROOT
SHADERS = p3.SHADERS
WMMA_RS = p3.WMMA_RS

C1 = {
    "source": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave_p3_c1.comp",
    "binary": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave_p3_c1.spv",
    "source_sha256": "9857a8ad457f3b119e0465dde7c4bb36635053e9ee25522bce0a208452953da4",
    "binary_sha256": "b072242034ba7bd0f53690c38c03b11ed587b538480c04ed13ecf19fe2a8b27d",
    "source_bytes": 26_229,
    "binary_bytes": 36_912,
    "shader_name": "wmma_gemm_f16_f16acc_twowave_p3_c1",
}

# Tokens the predicate must still contain. Each one is a condition, not a
# comment: a predicate that loses one of them admits a geometry it cannot prove.
PREDICATE_TOKENS = (
    "geometry.rows % 32 != 0",
    "geometry.columns % 32 != 0",
    "geometry.reduction % 16 != 0",
    "lda % 2 != 0",
    "ldb % 2 != 0",
    "u32::MAX as u64",
    "if a_last > ceiling || b_last > ceiling {",
    "a_capacity_bytes",
    "b_capacity_bytes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def check_c1_artifacts(root: Path) -> None:
    source = root / C1["source"]
    binary = root / C1["binary"]
    require(source.is_file(), f"C1: shader source is missing ({C1['source']})")
    require(binary.is_file(), f"C1: SPIR-V is missing ({C1['binary']})")
    require(digest(source) == C1["source_sha256"], "C1: the shader source is not the recovered bytes")
    require(digest(binary) == C1["binary_sha256"], "C1: the SPIR-V is not the recovered bytes")
    require(source.stat().st_size == C1["source_bytes"], "C1: shader source size changed")
    require(binary.stat().st_size == C1["binary_bytes"], "C1: SPIR-V size changed")
    summary = p3.spirv_summary(binary)
    require(summary["version"] == p3.P3["spirv_version"], "C1: SPIR-V version differs from P3's")
    # C1 is a staging change on the same arithmetic, so it must issue the same
    # cooperative operations: four multiply-adds over three cooperative types.
    require(
        summary["cooperative_mul_adds"] == p3.P3["cooperative_mul_adds"],
        "C1: cooperative multiply-add count differs from the P3 contract",
    )
    require(
        summary["cooperative_types"] == p3.P3["cooperative_types"],
        "C1: cooperative type count differs from the P3 contract",
    )


def check_c1_embedding(root: Path) -> None:
    text = read(root / WMMA_RS)
    require(
        'include_bytes!("../shaders/wmma_gemm_f16_f16acc_twowave_p3_c1.spv")' in text,
        "the C1 module is not embedded from its committed artifact",
    )
    staged = p3.function_body(text, "pub fn staged_spirv(self) -> Option<&'static [u8]>")
    require(
        "Self::P3 => Some(WMMA_GEMM_F16ACC_TWOWAVE_P3_C1_SPV)" in staged,
        "FullFp16Variant::P3 no longer reaches the C1 module",
    )
    require(
        "Self::Split8 | Self::TwoBank | Self::TwoWave => None," in staged
        and staged.count("=> None,") == 1
        and staged.count("Some(") == 1,
        "a non-P3 variant was given a batched-staging module",
    )
    names = p3.function_body(text, "pub fn staged_shader_name(self) -> Option<&'static str>")
    require(
        f'Self::P3 => Some("{C1["shader_name"]}")' in names,
        "the batched-staging shader name changed",
    )
    require(
        "Self::Split8 | Self::TwoBank | Self::TwoWave => None," in names
        and names.count("=> None,") == 1,
        "a non-P3 variant reports a batched-staging name",
    )
    # The contract module is still what the variant's `spirv()` and shader name
    # report; the P3 gate above already pinned both to the v0.2.5 bytes.
    require(
        "Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV," in p3.function_body(
            text, "pub fn spirv(self) -> &'static [u8]"
        ),
        "the P3 contract module is no longer the variant's reference module",
    )


def check_staging_geometry(root: Path) -> None:
    text = read(root / WMMA_RS)
    lda = p3.function_body(text, "pub fn lda(&self) -> usize")
    ldb = p3.function_body(text, "pub fn ldb(&self) -> usize")
    require("self.layout.transposed_a" in lda and "self.rows" in lda and "self.reduction" in lda,
            "StagingGeometry::lda no longer matches the dispatch's derived stride")
    require("self.layout.transposed_b" in ldb and "self.columns" in ldb and "self.reduction" in ldb,
            "StagingGeometry::ldb no longer matches the dispatch's derived stride")
    record = p3.function_body(text, "pub fn record_gemm(")
    require(
        record.count("batched_staging_eligible(&StagingGeometry {") == 1,
        "the production selector does not evaluate the staging contract exactly once",
    )
    require(
        "a_capacity_bytes: a.capacity_bytes()" in record
        and "b_capacity_bytes: b.capacity_bytes()" in record,
        "the staging contract is not evaluated against the bound operands' capacities",
    )
    require(
        "WmmaGemmArm::FullFp16Accumulate => {" in record,
        "the staging selection is no longer confined to the Full FP16 arm",
    )
    require(
        "_ => masked," in record,
        "a geometry outside the contract no longer falls back to the original P3 module",
    )
    # The counters: one increment per route, inside the Full FP16 arm.
    require(
        record.count("self.staging_batched.fetch_add(1, Ordering::Relaxed);") == 1
        and record.count("self.staging_masked.fetch_add(1, Ordering::Relaxed);") == 1,
        "the staging dispatch counters are not incremented exactly once per route",
    )
    counter_guard = record.find("if arm == WmmaGemmArm::FullFp16Accumulate {")
    require(counter_guard >= 0, "a staging counter is moved outside the Full FP16 arm")
    require(
        record.find("self.staging_batched.fetch_add") > counter_guard
        and record.find("self.staging_masked.fetch_add") > counter_guard,
        "a staging counter is moved outside the Full FP16 arm",
    )
    require(
        "pub fn staging_counts(&self) -> WmmaStagingCounts" in text,
        "the per-dispatch staging receipt accessor was removed",
    )


def check_predicate(root: Path) -> None:
    text = read(root / WMMA_RS)
    body = p3.function_body(text, "pub fn batched_staging_eligible(")
    missing = [token for token in PREDICATE_TOKENS if token not in body]
    require(not missing, "the staging predicate lost a condition: " + ", ".join(missing))
    require(
        body.count("return false;") >= 4 and body.count("None => return false,") == 2,
        "the staging predicate returns early fewer times than it has conditions",
    )
    require("if rows == 0 || columns == 0 || reduction == 0 {" in body,
            "the staging predicate no longer rejects an empty geometry")
    require("checked_mul(2)" in body,
            "the staging predicate no longer checks the operand byte extent for overflow")
    # A predicate that cannot be reached from production is not a selector, and
    # one that is reachable from off/mixed arms would move the Mixed route.
    require(
        read(root / WMMA_RS).count("batched_staging_eligible(") >= 2,
        "the staging predicate is declared but never consulted",
    )


def check_no_forced_route(root: Path) -> None:
    text = read(root / WMMA_RS)
    # No new knob: the only variant selector stays the v0.2.5 environment
    # variable, and no environment variable or CLI flag can name C1.
    require(
        'pub const FULL_FP16_VARIANT_ENV: &str = "HIERARCHOS_WMMA_FULL_FP16_VARIANT";' in text,
        "the Full FP16 variant environment contract changed",
    )
    for fragment in ("HIERARCHOS_WMMA_C1", "HIERARCHOS_WMMA_STAGING", "staged_shader_name()?"):
        require(fragment not in text, f"a forced staging route appeared: {fragment}")
    parse = p3.function_body(text, "pub fn parse(value: &str) -> Result<Self>",
                             after="impl FullFp16Variant {")
    require("c1" not in parse.lower() and "batch" not in parse.lower(),
            "the variant parser exposes a staging implementation as a selectable variant")
    require("from_env" in text and "staged_spirv" not in p3.function_body(
        text, "pub fn from_env_value(value: Option<String>)"),
        "the staging module is reachable from the environment contract")


def check_source_structure(root: Path) -> None:
    """The pinned C1 source keeps the barrier structure it was qualified with."""
    c1 = read(root / C1["source"])
    p3_source = read(root / p3.P3["source"])
    require(c1.count("barrier();") == p3_source.count("barrier();"),
            "the C1 source changed the barrier count relative to P3")
    # Exactly four fast paths were *added*; the masked branches they fall back
    # to are P3's, unchanged.
    masked_p3 = p3_source.count("if (wide_a &&") + p3_source.count("if (wide_b &&")
    require(
        c1.count("if (wide_a &&") + c1.count("if (wide_b &&") == masked_p3 + 4,
        "the C1 source no longer has exactly four added uniform staging fast paths",
    )
    require(c1.count("wide_a && m0 + 32u <= pc.m && k0 + 16u <= pc.k") +
            c1.count("wide_a && k0 + 16u <= pc.k && m0 + 32u <= pc.m") +
            c1.count("wide_b && k0 + 16u <= pc.k && n0 + 32u <= pc.n") +
            c1.count("wide_b && n0 + 32u <= pc.n && k0 + 16u <= pc.k") == 4,
            "a staging fast-path predicate changed: it must be the uniform bounds+parity test")
    # The fast path must stay inside the C1-only artifact, never in P3.
    require("wide_a && m0 + 32u <= pc.m" not in p3_source,
            "the batched staging path leaked into the v0.2.5 P3 module")


def validate(root: Path = ROOT) -> dict:
    root = Path(root)
    # The v0.2.5 gate first, unmodified: this release may not weaken it.
    baseline = p3.validate(root)
    check_c1_artifacts(root)
    check_c1_embedding(root)
    check_staging_geometry(root)
    check_predicate(root)
    check_no_forced_route(root)
    check_source_structure(root)
    return {"baseline": baseline, "c1": {k: C1[k] for k in ("source_sha256", "binary_sha256",
                                                           "binary_bytes", "shader_name")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.root)
    except (ValueError, KeyError, FileNotFoundError, UnicodeDecodeError) as error:
        print(f"PHASE LXXXVIII C1 RELEASE VALIDATION: FAIL ({error})")
        return 1
    if args.json:
        import json
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    print(
        "PHASE LXXXVIII C1 RELEASE VALIDATION: PASS "
        "(the unmodified v0.2.5 P3 gate passes; the C1 staging source and SPIR-V are "
        f"the recovered bytes {C1['binary_bytes']:,} B; only FullFp16Variant::P3 can "
        "reach the batched-staging module; the geometry contract keeps the tile, "
        "packed-pair parity and 32-bit bound conditions and checks both bound "
        "operands' capacities; every non-proven geometry dispatches the original "
        "P3 module; no environment variable, flag or variant name can force the "
        "staging route; the C1 source keeps P3's barrier count and exactly four "
        "uniform fast-path predicates; the GPU qualification is the recorded "
        "Phase LXXXVIII evidence, and hosted CI executed no adapter)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

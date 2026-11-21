#!/usr/bin/env python3
"""Current-release gate for the Phase LXXXI P3 production Full FP16 arm.

`check_current_candidate.py` belongs to **Phase LXIV**: it holds that phase's
implementation to its recorded base, its fixed protected-path inventory and its
own eight-bank split-reduction receipt. It is history, and it is run that way -
against the revision Phase LXIV was qualified at, in a detached worktree (see
`.github/workflows/candidate-a.yml`). Asking it to grade the cumulative Phase
LXXXI release tree as if that tree were the Phase LXIV change cannot work: the
release legitimately adds the two-wave and P3 modules it gates, so every one of
them reads as "protected source changed outside the qualified paths".

This gate asks the complementary question about the tree under review, and it is
deliberately narrower than a path allowlist. It pins **bytes and structure**:

* the P3 shader source and SPIR-V are the exact qualified artifacts
  (`4e448e3c…` / `2aff4123…`, 23,280 B, SPIR-V 1.5, 4 cooperative
  multiply-adds over 3 cooperative types);
* `wmma.rs` embeds that exact module and `FullFp16Variant::P3` is the variant
  the production selector resolves - by `include_bytes!`, by `spirv()` and by
  `shader_name()`, so the runtime pipeline identity cannot name a different
  artifact;
* `FullFp16Variant` has exactly the four committed variants and its parser has
  exactly four arms, so the production selector cannot reach a
  qualification-only module;
* `WmmaMode::Off` is still the external default (`HIERARCHOS_WMMA_MODE` unset or
  empty), so WMMA remains opt-in;
* the 16,000,000-MAC profitability predicate is unchanged and is applied at
  every production dispatch admission point - `ProjectionWmma::admits` (which
  gates construction) and its three recorded-batch call sites, and the
  Transformer graph's `wmma_linear`;
* unsupported and unprofitable geometries still fall back to the plain
  FP16-storage / FP32 kernels rather than dispatching cooperatively;
* the Phase LXII Mixed module and its production route are byte-identical;
* the qualified split8, two-bank and two-wave fallbacks are still committed,
  still selectable, and still dispatch their own modules.

The GPU qualification is the Phase LXXIX/LXXXI hardware evidence in
`phase_lxxxi_evidence/` and `PHASE_LXXXI_RELEASE_REPORT.md`. This gate is
CPU-side and never claims to have executed an RDNA3 adapter, produced an RGP
capture or measured a millisecond.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SPIRV_MAGIC = 0x07230203
OP_TYPE_COOPERATIVE_MATRIX = 4456
OP_COOPERATIVE_MATRIX_MUL_ADD = 4459

SHADERS = "hierarchos-vulkan/shaders"
WMMA_RS = "hierarchos-vulkan/src/wmma.rs"
PROJECTION_RS = "hierarchos-vulkan/src/projection.rs"
TRANSFORMER_RS = "hierarchos-vulkan/src/transformer.rs"

MAC_FLOOR = 16_000_000

# The Phase LXXXI release artifacts. These are the two hashes every Phase
# LXXIX/LXXXI receipt, the release report and the SHA-256 manifest agree on.
P3 = {
    "source": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave_p3.comp",
    "binary": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave_p3.spv",
    "source_sha256": "4e448e3cc6d5cccb1a39753f43dcc66d9c287a29a06dc9b4957224271b786d12",
    "binary_sha256": "2aff4123544757b0d3aaf8eb1fe967a0c00c692eb73a74e49af802d7b5c517d2",
    "source_bytes": 17_941,
    "binary_bytes": 23_280,
    "cooperative_mul_adds": 4,
    "cooperative_types": 3,
    "spirv_version": 0x00010500,
    "shader_name": "wmma_gemm_f16_f16acc_twowave_p3",
    "variant": "P3",
}

# The Phase LXII Mixed module. The release must not move it: it is the measured
# production Mixed route and its bytes are also pinned by the Phase LXV record.
MIXED = {
    "source": f"{SHADERS}/wmma_gemm_f16_f32acc.comp",
    "binary": f"{SHADERS}/wmma_gemm_f16_f32acc.spv",
    "source_sha256": "239b63144bdd7c7aef703d88622201bc9c6c4832000e7dc86cec3317034863fc",
    "binary_sha256": "e07e48882a2459ebe30be228675596a12ac3f6b846badf32099ef06cca5cea65",
    "cooperative_mul_adds": 4,
    "shader_name": "wmma_gemm_f16_f32acc",
}

# The earlier Full FP16 modules that stay committed and selectable. `split8` is
# the Phase LXIV eight-bank module; `two-bank` is Phase LXIX and `two-wave` is
# Phase LXXI. The release promotes P3 without removing any of them.
FALLBACKS = {
    "split8": {
        "source": f"{SHADERS}/wmma_gemm_f16_f16acc.comp",
        "binary": f"{SHADERS}/wmma_gemm_f16_f16acc.spv",
        "binary_sha256": "dbed2a528f674a37fd6672ff83a786eb28804a77a85befd076acc80d701106fe",
        "shader_name": "wmma_gemm_f16_f16acc",
        "variant": "Split8",
    },
    "two-bank": {
        "source": f"{SHADERS}/wmma_gemm_f16_f16acc_twobank.comp",
        "binary": f"{SHADERS}/wmma_gemm_f16_f16acc_twobank.spv",
        "binary_sha256": "169ff4861c813999deb90780a2bb6f78b9a2169fe592d050b57051a1fb42d6d3",
        "shader_name": "wmma_gemm_f16_f16acc_twobank",
        "variant": "TwoBank",
    },
    "two-wave": {
        "source": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave.comp",
        "binary": f"{SHADERS}/wmma_gemm_f16_f16acc_twowave.spv",
        "binary_sha256": "1b007264f96b6fd9d6fc01ddac939153829956606a428a64dd7a37ccd1948c96",
        "shader_name": "wmma_gemm_f16_f16acc_twowave",
        "variant": "TwoWave",
    },
}

# The whole-variant inventory. Exactly these four committed modules may be
# dispatched; there is no fifth "qualification" entry the selector could reach.
VARIANT_INVENTORY = {"Split8", "TwoBank", "TwoWave", "P3"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def function_body(text: str, signature: str, after: str | None = None) -> str:
    """The braced body of the first function whose signature contains ``signature``.

    ``after`` restricts the search to the part of the file following that marker.
    Both `WmmaMode` and `FullFp16Variant` declare `fn parse(&str)`, so the
    enclosing `impl` block has to disambiguate them.
    """
    start = text.find(after) if after else 0
    require(start >= 0, f"missing source construct: {after}")
    found = text.find(signature, start)
    require(found >= 0, f"missing source construct: {signature.split('(')[0].strip()}")
    opened = text.find("{", found)
    require(opened >= 0, f"missing body for source construct: {signature.split('(')[0].strip()}")
    depth = 0
    for index in range(opened, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opened:index + 1]
    raise ValueError(f"unbalanced body for source construct: {signature}")


def enum_variants(body: str) -> list[str]:
    """Variant names declared in an enum body, ignoring comments and attributes."""
    names = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith("{"):
            continue
        match = re.fullmatch(r"([A-Z][A-Za-z0-9]*),", line)
        if match:
            names.append(match.group(1))
    return names


def spirv_summary(path: Path) -> dict:
    data = Path(path).read_bytes()
    require(len(data) >= 20 and len(data) % 4 == 0,
            f"{path.name} is not a whole number of 32-bit words")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    require(words[0] == SPIRV_MAGIC, f"{path.name} is not a SPIR-V module")
    counts: dict[int, int] = {}
    index = 5
    while index < len(words):
        length = words[index] >> 16
        require(length > 0, f"{path.name} has a zero-length instruction")
        require(index + length <= len(words),
                f"{path.name} instruction stream overruns the module")
        opcode = words[index] & 0xFFFF
        counts[opcode] = counts.get(opcode, 0) + 1
        index += length
    require(index == len(words), f"{path.name} instruction stream does not tile the module")
    return {
        "version": words[1],
        "cooperative_mul_adds": counts.get(OP_COOPERATIVE_MATRIX_MUL_ADD, 0),
        "cooperative_types": counts.get(OP_TYPE_COOPERATIVE_MATRIX, 0),
    }


def check_artifacts(root: Path) -> dict:
    """Pin the P3 artifact and the two modules whose route must not move."""
    for label, spec in (("P3", P3), ("Mixed", MIXED)):
        source, binary = root / spec["source"], root / spec["binary"]
        require(source.is_file(), f"{label}: shader source is missing ({spec['source']})")
        require(binary.is_file(), f"{label}: SPIR-V is missing ({spec['binary']})")
        require(digest(source) == spec["source_sha256"],
                f"{label}: shader source is not the qualified bytes")
        require(digest(binary) == spec["binary_sha256"],
                f"{label}: SPIR-V is not the qualified bytes")
    require((root / P3["source"]).stat().st_size == P3["source_bytes"],
            "P3: shader source size changed")
    require((root / P3["binary"]).stat().st_size == P3["binary_bytes"],
            "P3: SPIR-V size changed")
    summary = spirv_summary(root / P3["binary"])
    require(summary["version"] == P3["spirv_version"],
            f"P3: SPIR-V version is {summary['version']:#x}, not {P3['spirv_version']:#x}")
    require(summary["cooperative_types"] == P3["cooperative_types"],
            f"P3: cooperative matrix type count is {summary['cooperative_types']}, "
            f"not {P3['cooperative_types']}")
    require(summary["cooperative_mul_adds"] == P3["cooperative_mul_adds"],
            f"P3: cooperative multiply-add count is {summary['cooperative_mul_adds']}, "
            f"not {P3['cooperative_mul_adds']}")
    require(spirv_summary(root / MIXED["binary"])["cooperative_mul_adds"]
            == MIXED["cooperative_mul_adds"],
            "Mixed: cooperative multiply-add count changed")
    return summary


def check_runtime_identity(root: Path) -> None:
    """The embedded module, the variant and the reported identity agree."""
    text = read(root / WMMA_RS)
    include = (f'const WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV: &[u8] =\n'
               f'    include_bytes!("../shaders/{P3["binary"].rsplit("/", 1)[-1]}");')
    require(include in text,
            "the P3 module is no longer embedded from the committed SPIR-V artifact")
    require("Self::P3 => WMMA_GEMM_F16ACC_TWOWAVE_P3_SPV,"
            in function_body(text, "pub fn spirv(self) -> &'static [u8]"),
            "FullFp16Variant::P3 does not dispatch the embedded P3 module")
    require(f'Self::P3 => "{P3["shader_name"]}",'
            in function_body(text, "pub fn shader_name(self) -> &'static str"),
            "FullFp16Variant::P3 no longer reports the qualified shader name")
    # The pipeline identity is built from the loaded kernel's shader name and the
    # selected variant's own SPIR-V length, so a P3 selection cannot report a
    # different module's identity.
    identity = function_body(text, "pub fn full_fp16_pipeline_identity(&self) -> Option<String>")
    require("kernel.shader_name().unwrap_or(\"unknown\")" in identity
            and "kernel.shader_signature()" in identity
            and "self.full_fp16_variant.spirv().len()" in identity,
            "the reported pipeline identity no longer derives from the loaded module")
    require("There is no\n    /// caller-supplied or qualification-only module in this build"
            in text,
            "the runtime identity contract that excludes qualification-only modules changed")


def check_selector(root: Path) -> None:
    """P3 is the default Full FP16 variant, and no fifth variant is reachable."""
    text = read(root / WMMA_RS)
    body = function_body(text, "pub enum FullFp16Variant")
    variants = enum_variants(body)
    require(variants == ["Split8", "TwoBank", "TwoWave", "P3"],
            "the Full FP16 variant inventory changed: " + ", ".join(variants))
    require(set(variants) == VARIANT_INVENTORY,
            "an unexpected Full FP16 variant is selectable")
    require("#[default]\n    P3," in text,
            "FullFp16Variant no longer defaults to the production P3 module")
    require("pub const ALL: [Self; 4] = [Self::Split8, Self::TwoBank, Self::TwoWave, Self::P3];"
            in text,
            "the committed Full FP16 variant inventory changed")
    parse = function_body(text, "pub fn parse(value: &str) -> Result<Self>",
                          after="impl FullFp16Variant {")
    arms = re.findall(r"=>\s*Ok\(Self::([A-Za-z0-9]+)\)", parse)
    require(sorted(arms) == sorted(VARIANT_INVENTORY),
            "the Full FP16 parser no longer resolves exactly the committed variants")
    require(len(re.findall(r"Ok\(Self::", parse)) == len(VARIANT_INVENTORY),
            "the Full FP16 parser has an ambiguous or duplicated arm")
    require("other => bail!(" in parse,
            "the Full FP16 parser no longer rejects unknown variant names")
    require('assert!(FullFp16Variant::from_env_value(Some("experimental".to_owned())).is_err());'
            in text,
            "the explicit rejection of qualification-only variant names is gone")
    # An absent or empty environment value takes the enum's own default.
    require("_ => Ok(Self::default()),"
            in function_body(text, "pub fn from_env_value(value: Option<String>)"),
            "FullFp16Variant::from_env_value no longer defers to the default variant")


def check_opt_in_mode(root: Path) -> None:
    """WMMA stays opt-in: an unset or empty mode is still `off`."""
    text = read(root / WMMA_RS)
    require(enum_variants(function_body(text, "pub enum WmmaMode"))
            == ["Off", "Mixed", "FullFp16"],
            "the WmmaMode inventory changed")
    require("#[default]\n    Off," in text,
            "WmmaMode no longer defaults to off")
    require("_ => Ok(Self::Off),"
            in function_body(text, "pub fn from_env() -> Result<Self>"),
            "an unset HIERARCHOS_WMMA_MODE no longer resolves to off")
    require("Off => \"off\"," in function_body(text, "pub fn label(self) -> &'static str"),
            "the external off label changed")


def check_policy(root: Path) -> None:
    """The 16,000,000-MAC threshold and the eligibility gate are unchanged."""
    text = read(root / WMMA_RS)
    require(f"pub const WMMA_MIN_PROFITABLE_MACS: u64 = {MAC_FLOOR:_};" in text,
            "the 16,000,000-MAC threshold moved")
    predicate = function_body(text, "pub fn wmma_shape_profitable(")
    require("wmma_shape_eligible(rows, columns, reduction)" in predicate,
            "the profitability predicate stopped requiring shape eligibility")
    require(">= WMMA_MIN_PROFITABLE_MACS" in predicate,
            "the profitability predicate stopped applying the measured floor")
    require("saturating_mul(reduction as u64)" in predicate,
            "the MAC product is no longer saturating, so an absurd shape could wrap")
    eligible = function_body(text, "pub fn wmma_shape_eligible(")
    for fragment in ("rows >= 16", "columns >= 16", "reduction >= 16",
                     "rows.saturating_mul(columns) >= 1024",
                     "rows.div_ceil(32).saturating_mul(columns.div_ceil(32)) >= 4"):
        require(fragment in eligible, f"the shape eligibility gate changed: {fragment}")
    require("assert!(WMMA_MIN_PROFITABLE_MACS > 128 * 256 * 256);" in text,
            "the profitability floor sanity assertion is gone")
    # Unsupported geometry keeps its safe route instead of dispatching WMMA.
    select = function_body(text, "pub fn select_arm(")
    require(select.count("Some(WmmaGemmArm::PlainF16Storage)") == 2,
            "an ineligible shape no longer falls back to the plain kernel")
    require("WmmaMode::Off => None," in select,
            "an off request still resolves to a cooperative arm")


def check_dispatch_admission(root: Path) -> None:
    """Every production admission point applies the profitability predicate."""
    projection = read(root / PROJECTION_RS)
    admits = function_body(projection, "pub(crate) fn admits(max_rows: usize")
    require("wmma::wmma_shape_profitable(max_rows, output_dim, input_dim)" in admits,
            "ProjectionWmma::admits no longer applies the profitability predicate")
    require("mode if !ProjectionWmma::admits(max_rows, input_dim, output_dim) =>"
            in projection,
            "the projection constructor no longer gates the cooperative executor")
    require(projection.count(
        ".filter(|_| ProjectionWmma::admits(rows, self.input_dim, self.output_dim))") == 3,
        "a projection forward/weight-gradient/input-gradient path lost its admission gate")
    require("crate::wmma_linear::note_policy_fallback();" in projection,
            "a declined projection no longer records its FP32 fallback")

    transformer = read(root / TRANSFORMER_RS)
    require("if !crate::wmma::wmma_shape_profitable(rows, output_dim, input_dim) {"
            in transformer,
            "the Transformer graph no longer applies the profitability predicate")
    require("crate::wmma_linear::note_policy_fallback();" in transformer,
            "a declined Transformer projection no longer records its FP32 fallback")
    graph = transformer.find(
        "if !crate::wmma::wmma_shape_profitable(rows, output_dim, input_dim) {")
    tail = transformer[graph:graph + 900]
    require("return Ok(None);" in tail,
            "a declined Transformer projection no longer stays on its FP32 kernel")


def check_mixed_route(root: Path) -> None:
    """The Mixed module and the production Mixed route are unchanged."""
    text = read(root / WMMA_RS)
    mixed_include = ('const WMMA_GEMM_F32ACC_SPV: &[u8] = include_bytes!("../shaders/'
                     + MIXED["binary"].rsplit("/", 1)[-1] + '");')
    require(mixed_include in text,
            "the production Mixed module is no longer embedded from its artifact")
    require('Some(gemm_kernel(WMMA_GEMM_F32ACC_SPV, "mixed")?)' in text,
            "the Mixed arm no longer builds its kernel from the qualified module")
    require("WmmaGemmArm::MixedFp32Accumulate" in text,
            "the Mixed accumulation route changed")
    for limit in ("pub const MIXED_VS_FP32_MAX_ABS: f64 = 2.0e-2;",
                  "pub const MIXED_VS_FP32_RMS: f64 = 3.0e-3;",
                  "pub const MIXED_VS_MATCHED_ORACLE_MAX_ABS: f64 = 1.0e-3;"):
        require(limit in text, f"a Mixed numerical limit changed: {limit}")


def check_fallbacks(root: Path) -> None:
    """The qualified earlier Full FP16 modules are still committed and selectable."""
    text = read(root / WMMA_RS)
    spirv = function_body(text, "pub fn spirv(self) -> &'static [u8]")
    for label, spec in FALLBACKS.items():
        for key in ("source", "binary"):
            require((root / spec[key]).is_file(),
                    f"{label}: committed artifact is missing ({spec[key]})")
        require(digest(root / spec["binary"]) == spec["binary_sha256"],
                f"{label}: the committed SPIR-V is not the qualified bytes")
        require(f'include_bytes!("../shaders/{spec["binary"].rsplit("/", 1)[-1]}");' in text,
                f"{label}: the module is no longer embedded")
        require(f"Self::{spec['variant']} => WMMA_GEMM_F16ACC_"
                in spirv,
                f"{label}: the variant no longer dispatches its own module")
        require(f'Self::{spec["variant"]} => "{spec["shader_name"]}",' in text,
                f"{label}: the variant no longer reports its own shader name")
    require("HIERARCHOS_WMMA_FULL_FP16_VARIANT" in text,
            "the fallback selection environment variable was removed")


def validate(root: Path = ROOT) -> dict:
    root = Path(root)
    require((root / "hierarchos-vulkan").is_dir(), "not a repository root")
    summary = check_artifacts(root)
    check_runtime_identity(root)
    check_selector(root)
    check_opt_in_mode(root)
    check_policy(root)
    check_dispatch_admission(root)
    check_mixed_route(root)
    check_fallbacks(root)
    return {
        "p3_spirv_version": f"{summary['version']:#010x}",
        "p3_cooperative_mul_adds": summary["cooperative_mul_adds"],
        "p3_cooperative_types": summary["cooperative_types"],
        "mac_floor": MAC_FLOOR,
        "variants": sorted(VARIANT_INVENTORY),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.root)
    except (ValueError, KeyError, FileNotFoundError, UnicodeDecodeError) as error:
        print(f"PHASE LXXXI P3 RELEASE VALIDATION: FAIL ({error})")
        return 1
    if args.json:
        import json
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    print(
        "PHASE LXXXI P3 RELEASE VALIDATION: PASS "
        f"(P3 source and SPIR-V are the qualified bytes, {P3['binary_bytes']:,} B, "
        f"SPIR-V {result['p3_spirv_version']}, "
        f"{result['p3_cooperative_mul_adds']} cooperative multiply-adds over "
        f"{result['p3_cooperative_types']} cooperative types, "
        f"{P3['variant']} is the default Full FP16 variant and the production "
        f"selector reaches only the {len(VARIANT_INVENTORY)} committed variants, "
        f"WMMA stays opt-in with off as the external default, the "
        f"{MAC_FLOOR:,}-MAC profitability predicate gates every production "
        "admission point, unsupported and unprofitable shapes keep their plain/FP32 "
        "kernels, the Mixed module and route are unchanged, and the split8, two-bank "
        "and two-wave fallbacks remain committed and selectable; the GPU "
        "qualification is the recorded Phase LXXIX/LXXXI RDNA3 evidence, and "
        "hosted CI executed no adapter)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

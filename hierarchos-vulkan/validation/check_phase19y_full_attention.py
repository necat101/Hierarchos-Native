#!/usr/bin/env python3
"""Fail-closed verifier for the Phase XIX-Y full-attention receipt.

The receipt is a current-source execution receipt: it binds the 16 expanded
GPT-2 shape cases and the 74 shared retained cases to the current source,
binary, CPU oracle manifest and fixture bytes.  Historical Phase XVI
architecture-specific entries are accounted for separately and are never
reported as current passing comparisons.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parent
RECEIPT = (
    CRATE / "validation/release_parity/multimodal_training"
    / "20261112-phase19y-full-attention" / "current-qualification.json"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
LIMIT = 2e-7
SHAPES = [(2, 8, 4), (1, 8, 4), (3, 8, 4), (2, 6, 4),
          (2, 10, 4), (2, 16, 4), (2, 8, 2), (2, 8, 8)]
SHAPE_PROFILES = {f"gpt2_b{b}_s{s}_h{h}" for b, s, h in SHAPES}
SHAPE_CASES = {f"{profile}/{mask}" for profile in SHAPE_PROFILES
               for mask in ("unmasked", "padded")}
SHARED_PROFILES = {
    "gpt2", "llama", "smollm3", "smollm3_yarn", "gpt_oss", "qwen2",
    "qwen2_5_gqa", "qwen2_5_sliding_tied", "mistral", "phi3",
    "phi4_multimodal_text", "gemma3", "gemma3_softcap_tied", "gemma4",
    "kimi_k25_text", "mistral4", "deepseek_v4", "mixtral", "qwen3_5_full",
    "qwen3_5_linear", "qwen3_5_mixed", "qwen3_5_moe", "qwen4_exp_linear",
    "qwen4_exp_qsa", "qwen4_exp_ple", "qwen4_exp_mixed_ple", "minimax_m2",
    "minimax_m3_dense", "minimax_m3", "dbrx", "bert", "gpt2_wide",
    "llama_wide", "falcon_h1_gated_after", "falcon_h1_gated_before",
    "falcon_h1_silu", "qwen3_next",
}
SHARED_CASES = {f"{profile}/{mask}" for profile in SHARED_PROFILES
                for mask in ("unmasked", "padded")}
EXPECTED_MODEL = {
    "config.json": "6d1760e9facce5ae3913a1ed60fc7f8fe29e35eb7b662be49861f9e87cd2aa89",
    "image/config.json": "1f87da2a0a6c49de7d3d0f265e98d31b206de80f0d70b584369153a6c41584f1",
    "image/model.safetensors": "aed8d82b2f07615ae2e0c5f15b9794c2208138d802fd1ae26017d0e19437ffd1",
    "image/preprocessor_config.json": "effa96347b7c954a56e7c2fce8aa0b0f1c24d161a947d33f5df137d15af13231",
    "text/config.json": "aa844a8565ae1a30991e5c736533d110cdc6e58462a72f5ee1e5f9eee6f3986e",
    "text/generation_config.json": "ac3289fe14c15c83e32d983a6b1c44d29ff1e90ef7b9393bc7f155bc0e9a4561",
    "text/model.safetensors": "bf2ba7751555a1921abbad2e249c4ddd5f5e1f2be9cde3ac6c0cff75a1077c93",
}
SHAPE_MANIFEST_SHA = "709427881624237c05dccf1e9bdd7593285819304af7d3778b885bc206fbad0c"
SHARED_MANIFEST_SHA = "a98dc5c32f4ba044b59a7a1e03fcac2b32d234ae12fde7af8ca3b1ff00cad1be"
SHAPE_ARCHIVE_SHA = "a45396b752d62b01f75aa0f0cc8a83ee71640bf44a9f83b841e7476056b6ef14"
SHAPE_ARCHIVE_BYTES = 32948229
SEALED_SHADER_SHA = "f8439dc2a5e94dd39d6e9bf3d537d87b9c25c734e0b216627ba7bafe57dd3226"
TORCH_REVISION = "2236df1770800ffea5697b11b0bb0d910b2e59e1"
TRANSFORMERS_REVISION = "129e8dc306932532e0b9e702d1fb4d5d7771c2a4"
BINARY_NAME = "image_text_pipeline_probe.exe"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def max_logit(rows):
    values = [check["max_abs"] for row in rows for check in row.get("checks", [])
              if check.get("metric", "").endswith("/logits")
              and check.get("max_abs") is not None]
    return max(values) if values else None


def validate(document: dict, receipt_dir: Path, crate: Path = CRATE,
             binary_root: Path | None = None, shape_fixtures: Path | None = None,
             shared_fixtures: Path | None = None,
             require_binary: bool = False, require_controls: bool = True) -> None:
    require(document.get("schema") == 1, "unexpected Phase XIX-Y schema")
    require(document.get("kind") == "phase19y-full-attention-current-source",
            "unexpected Phase XIX-Y kind")
    require(document.get("status") == "PASS" and document.get("pass") is True,
            "Phase XIX-Y receipt is not a pass")
    require(document.get("target") == "rdna3", "target changed")
    revision = str(document.get("source_revision", ""))
    require(HEX40.fullmatch(revision) is not None, "invalid source revision")
    if git("rev-parse", "HEAD") != revision:
        ancestor = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", revision, "HEAD"],
            capture_output=True,
        )
        require(ancestor.returncode == 0, "source revision is not an ancestor of HEAD")
    require(document.get("source_clean") is True, "receipt came from a dirty worktree")

    sources = document.get("source_files") or {}
    require(len(sources) >= 140, "Phase XIX-Y binds too few sources")
    for required in (
        "hierarchos-vulkan/src/transformer.rs",
        "hierarchos-vulkan/src/transformer/image_text.rs",
        "hierarchos-vulkan/src/transformer/vision.rs",
        "hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv",
    ):
        require(required in sources, "missing source binding: " + required)
    for name, expected in sources.items():
        require(SHA256.fullmatch(str(expected)) is not None, "invalid source digest")
        candidate = crate.parent / name
        require(candidate.is_file(), "missing bound source " + name)
        require(digest(candidate) == expected, "bound source changed: " + name)
    require(sources["hierarchos-vulkan/shaders/transformer_attention_forward_cpu_reciprocal.spv"]
            == SEALED_SHADER_SHA, "reciprocal shader is not the sealed artifact")

    binaries = document.get("binaries") or {}
    probe = binaries.get("image_text_pipeline_probe") or {}
    require(SHA256.fullmatch(str(probe.get("sha256"))) is not None,
            "missing or malformed probe binding")
    if binary_root is None:
        binary_root = crate / "target/debug"
    candidate = Path(binary_root) / BINARY_NAME
    if candidate.is_file():
        require(digest(candidate) == probe["sha256"], "probe binary changed")
    elif require_binary:
        raise ValueError("required probe binary is missing: " + str(candidate))

    oracle = document.get("oracle") or {}
    require(oracle.get("cpu") == "AVX512", "CPU oracle capability changed")
    require(oracle.get("cpu_capability") == "AVX512", "PyTorch AVX-512 path changed")
    require(oracle.get("torch") == "2.6.0+cpu", "CPU oracle torch changed")
    require(oracle.get("torch_revision") == TORCH_REVISION, "CPU oracle torch revision changed")
    require(oracle.get("threads") == 1, "CPU oracle thread count changed")
    require(oracle.get("transformers_revision") == TRANSFORMERS_REVISION,
            "CPU oracle transformers revision changed")
    require(oracle.get("input_model_sha256") == EXPECTED_MODEL,
            "CPU oracle input model changed")
    require(oracle.get("cpu_model") == "AMD Ryzen Z1 Extreme", "CPU model changed")
    require(SHA256.fullmatch(str(oracle.get("environment_sha256"))) is not None,
            "missing oracle environment binding")

    evidence = document.get("evidence") or {}
    require(evidence, "receipt has no evidence")
    for name, expected in evidence.items():
        require(SHA256.fullmatch(str(expected)) is not None, "invalid evidence digest")
        candidate = receipt_dir / name
        require(candidate.is_file(), "missing evidence " + name)
        require(digest(candidate) == expected, "evidence changed: " + name)

    fixtures = document.get("fixtures") or {}
    shape_fixture = fixtures.get("shapes") or {}
    shared_fixture = fixtures.get("shared") or {}
    require(shape_fixture.get("manifest_sha256") == SHAPE_MANIFEST_SHA,
            "shape fixture manifest hash changed")
    require(shape_fixture.get("archive_sha256") == SHAPE_ARCHIVE_SHA
            and shape_fixture.get("archive_bytes") == SHAPE_ARCHIVE_BYTES,
            "shape archive provenance changed")
    require(shared_fixture.get("manifest_sha256") == SHARED_MANIFEST_SHA,
            "shared fixture manifest hash changed")
    shape_manifest_path = receipt_dir / shape_fixture["manifest_path"]
    shared_manifest_path = receipt_dir / shared_fixture["manifest_path"]
    require(digest(shape_manifest_path) == SHAPE_MANIFEST_SHA,
            "shape manifest bytes changed")
    require(digest(shared_manifest_path) == SHARED_MANIFEST_SHA,
            "shared manifest bytes changed")
    shape_manifest = load(shape_manifest_path)
    shared_manifest = load(shared_manifest_path)
    require(shape_manifest.get("complete") is True, "shape manifest incomplete")
    require(len(shape_manifest.get("cases", [])) == 16, "shape manifest case count changed")
    require({row.get("name") for row in shape_manifest["cases"]} == SHAPE_CASES,
            "shape manifest coverage changed")
    require(shape_manifest.get("input_model_sha256") == EXPECTED_MODEL,
            "shape manifest input model changed")
    require(shared_manifest.get("complete") is True, "shared manifest incomplete")
    require(set(shared_manifest.get("required_green_profiles", [])) == SHARED_PROFILES,
            "shared profile inventory changed")
    shared_names = {row.get("name") for row in shared_manifest.get("cases", [])}
    require(SHARED_CASES <= shared_names, "shared manifest is missing a required case")

    def validate_results(rows, expected_cases, label):
        require(len(rows) == len(expected_cases), f"{label} coverage changed")
        names = [row.get("case") for row in rows]
        require(len(names) == len(set(names)), f"{label} has duplicate cases")
        require(set(names) == expected_cases, f"{label} case inventory changed")
        for row in rows:
            require(row.get("pass") is True, f"{label} case did not pass: {row.get('case')}")
            checks = row.get("checks") or []
            require(checks, f"{label} case has no checks: {row.get('case')}")
            for check in checks:
                require(check.get("pass") is True,
                        f"{label} check failed: {row.get('case')}/{check.get('metric')}")
                value = check.get("max_abs")
                if value is not None:
                    require(float(value) < LIMIT,
                            f"{label} threshold violated: {row.get('case')}/{check.get('metric')}")

    shapes = document.get("shapes") or {}
    require(shapes.get("cases") == 16 and shapes.get("expected_cases") == 16,
            "expanded shape count changed")
    require(shapes.get("pass") is True, "expanded shapes did not pass")
    require(shapes.get("binary_sha256") == probe["sha256"],
            "expanded shape binary binding changed")
    require(shapes.get("fixtures_sha256") == SHAPE_MANIFEST_SHA,
            "expanded shape fixture binding changed")
    shape_receipt_path = receipt_dir / shapes["receipt_path"]
    require(digest(shape_receipt_path) == shapes["receipt_sha256"],
            "expanded shape sub-receipt changed")
    shape_receipt = load(shape_receipt_path)
    require(shape_receipt.get("pass") is True
            and shape_receipt.get("completed_cases") == 16
            and shape_receipt.get("expected_cases") == 16,
            "expanded shape sub-receipt is incomplete")
    require(shape_receipt.get("binary_sha256") == probe["sha256"],
            "expanded shape sub-receipt binary changed")
    require(shape_receipt.get("fixtures_sha256") == SHAPE_MANIFEST_SHA,
            "expanded shape sub-receipt fixtures changed")
    require(shapes.get("results") == shape_receipt.get("results"),
            "expanded shape results differ from sub-receipt")
    validate_results(shapes.get("results") or [], SHAPE_CASES, "expanded shapes")
    shape_max = max_logit(shapes["results"])
    require(shape_max is not None and shape_max < LIMIT,
            "expanded shape maximum is missing or above the strict limit")
    require(shapes.get("max_abs") == shape_max, "expanded shape maximum is not re-derived")

    retained = document.get("retained") or {}
    shared = retained.get("shared_current") or {}
    require(shared.get("cases") == 74 and shared.get("profiles") == 37,
            "shared retained coverage changed")
    require(shared.get("pass") is True, "shared retained cases did not pass")
    require(shared.get("fixtures_sha256") == SHARED_MANIFEST_SHA,
            "shared retained fixture binding changed")
    shared_receipt_path = receipt_dir / shared["receipt_path"]
    require(digest(shared_receipt_path) == shared["receipt_sha256"],
            "shared retained sub-receipt changed")
    shared_receipt = load(shared_receipt_path)
    require(shared_receipt.get("pass") is True
            and shared_receipt.get("completed_cases") == 74
            and shared_receipt.get("expected_cases") == 74,
            "shared retained sub-receipt is incomplete")
    require(shared_receipt.get("fixtures_sha256") == SHARED_MANIFEST_SHA,
            "shared retained sub-receipt fixtures changed")
    require(shared_receipt.get("binary_sha256") == probe["sha256"],
            "shared retained sub-receipt binary changed")
    require(shared.get("results") == shared_receipt.get("results"),
            "shared retained results differ from sub-receipt")
    validate_results(shared.get("results") or [], SHARED_CASES, "shared retained")
    shared_max = max_logit(shared["results"])
    require(shared_max is not None and shared_max < LIMIT,
            "shared retained maximum is missing or above the strict limit")
    require(shared.get("max_abs") == shared_max, "shared retained maximum is not re-derived")

    architecture = retained.get("architecture_specific_historical") or {}
    require(architecture.get("total") == 19, "architecture-specific historical count changed")
    require(architecture.get("verified_current") is False,
            "architecture-specific historical entries are not current evidence")
    coverage = retained.get("coverage") or {}
    require(coverage.get("shared_current") == 74
            and coverage.get("architecture_specific_historical") == 19
            and coverage.get("total_accounted") == 93,
            "93-entry retained-family accounting changed")

    narrow = document.get("narrow_phase19x") or {}
    require(narrow.get("status") == "HISTORICAL_PRESERVED",
            "narrow Phase XIX-X status changed")
    require(narrow.get("cases") == 4 and narrow.get("comparisons") == 12
            and narrow.get("elements") == 3158064,
            "narrow Phase XIX-X coverage changed")
    require(narrow.get("pass") is True, "narrow Phase XIX-X receipt is not a pass")
    require(digest(receipt_dir / narrow["path"]) == narrow["receipt_sha256"],
            "narrow Phase XIX-X receipt changed")

    historical = document.get("historical_phase16") or {}
    require(historical.get("shapes") == 16 and historical.get("retained_entries") == 93,
            "historical Phase XVI accounting changed")
    require(historical.get("shared") == 74 and historical.get("architecture_specific") == 19,
            "historical Phase XVI family split changed")
    require(SHA256.fullmatch(str(historical.get("receipt_sha256"))) is not None,
            "historical Phase XVI receipt is not bound")

    shader = document.get("shader") or {}
    require(shader.get("sha256") == SEALED_SHADER_SHA,
            "restored reciprocal shader hash changed")
    require(shader.get("inventory") == 45 and shader.get("status_counts") == {"identical": 45},
            "45-artifact shader inventory changed")
    compatibility = document.get("phase19w") or {}
    require(compatibility.get("training_profiles") == 32
            and compatibility.get("generation_requests") == 72
            and compatibility.get("canonical_types") == 146
            and compatibility.get("aliases") == 83,
            "Phase XIX-W retained compatibility coverage changed")
    require(SHA256.fullmatch(str(compatibility.get("matrix_sha256"))) is not None,
            "Phase XIX-W matrix is not bound")

    controls = document.get("negative_controls") or {}
    if require_controls:
        require(controls.get("detected") == controls.get("total")
                and controls.get("total", 0) >= 12,
                "Phase XIX-Y negative controls incomplete")
        require(all(row.get("rejected") is True for row in controls.get("controls", [])),
                "a Phase XIX-Y negative control was not rejected")


def run_negative_controls(document: dict, receipt_dir: Path, crate: Path = CRATE,
                          binary_root: Path | None = None) -> dict:
    """Exercise the receipt's rejection paths and return the recorded result."""
    cases = []

    def reject(name, mutate):
        candidate = copy.deepcopy(document)
        mutate(candidate)
        try:
            validate(candidate, receipt_dir, crate, binary_root=binary_root,
                     require_binary=binary_root is not None, require_controls=False)
        except (ValueError, KeyError, FileNotFoundError):
            cases.append({"name": name, "rejected": True})
        else:
            cases.append({"name": name, "rejected": False})

    reject("stale source binding", lambda d: d["source_files"].__setitem__(
        "hierarchos-vulkan/src/transformer.rs", "0" * 64))
    reject("wrong source revision", lambda d: d.__setitem__("source_revision", "f" * 40))
    reject("wrong torch revision", lambda d: d["oracle"].__setitem__("torch_revision", "0" * 40))
    reject("wrong transformers revision", lambda d: d["oracle"].__setitem__(
        "transformers_revision", "0" * 40))
    reject("wrong model hash", lambda d: d["oracle"]["input_model_sha256"].__setitem__(
        "text/model.safetensors", "0" * 64))
    reject("wrong CPU capability", lambda d: d["oracle"].__setitem__("cpu_capability", "AVX2"))
    reject("wrong fixture manifest", lambda d: d["fixtures"]["shapes"].__setitem__(
        "manifest_sha256", "0" * 64))
    reject("wrong shared fixture manifest", lambda d: d["fixtures"]["shared"].__setitem__(
        "manifest_sha256", "0" * 64))
    reject("wrong archive provenance", lambda d: d["fixtures"]["shapes"].__setitem__(
        "archive_sha256", "0" * 64))
    reject("missing expanded case", lambda d: d["shapes"]["results"].pop())
    reject("duplicate shared case", lambda d: d["retained"]["shared_current"]["results"].append(
        copy.deepcopy(d["retained"]["shared_current"]["results"][0])))
    reject("strict threshold violation", lambda d: d["shapes"]["results"][0]["checks"][0].__setitem__(
        "max_abs", LIMIT))
    reject("failed check", lambda d: d["retained"]["shared_current"]["results"][0]["checks"][0].__setitem__(
        "pass", False))
    reject("incomplete 93 accounting", lambda d: d["retained"]["coverage"].__setitem__(
        "total_accounted", 92))
    reject("wrong narrow status", lambda d: d["narrow_phase19x"].__setitem__(
        "status", "CURRENT"))
    reject("wrong shader", lambda d: d["shader"].__setitem__("sha256", "0" * 64))
    reject("wrong matrix binding", lambda d: d["phase19w"].__setitem__("matrix_sha256", "0" * 64))
    reject("missing evidence", lambda d: d["evidence"].__setitem__("missing.json", "0" * 64))
    if binary_root is not None and (Path(binary_root) / BINARY_NAME).is_file():
        reject("wrong executable", lambda d: d["binaries"]["image_text_pipeline_probe"].__setitem__(
            "sha256", "0" * 64))
    return {"total": len(cases), "detected": sum(row["rejected"] for row in cases),
            "controls": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--binary-root", type=Path)
    parser.add_argument("--shape-fixtures", type=Path)
    parser.add_argument("--shared-fixtures", type=Path)
    parser.add_argument("--require-binary", action="store_true")
    args = parser.parse_args()
    try:
        document = load(args.receipt)
        validate(document, args.receipt.parent, binary_root=args.binary_root,
                 shape_fixtures=args.shape_fixtures, shared_fixtures=args.shared_fixtures,
                 require_binary=args.require_binary)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"PHASE XIX-Y FULL ATTENTION: FAIL ({error})")
        return 1
    print("PHASE XIX-Y FULL ATTENTION: PASS "
          f"({document['source_revision'][:12]}, 16/16 shapes, 74/74 shared, "
          f"{document['retained']['coverage']['total_accounted']} retained entries accounted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
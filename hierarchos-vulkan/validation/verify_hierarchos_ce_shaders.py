"""Reproduce the Hierarchos CE modules; --write explicitly updates artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def variants():
    for stem in (
        "cross_entropy_linear_input_grad_streaming",
        "cross_entropy_linear_input_grad_streaming_fp16_packed",
        "cross_entropy_linear_weight_grad_streaming",
        "cross_entropy_linear_input_grad_tape_fp16_packed",
        "cross_entropy_logits_to_grad_inplace",
        "cross_entropy_linear_row_stats_streaming",
        "cross_entropy_linear_row_stats_streaming_fp16_packed",
        "cross_entropy_row_stats_tape",
        "cross_entropy_row_stats_tile_partials",
    ):
        yield stem, stem, []
    stem = "cross_entropy_linear_row_stats_streaming_fp16_packed"
    yield stem + "_tape", stem, ["LM_CE_WRITE_LOGIT_TAPE=1"]
    stem = "cross_entropy_row_stats_tile_partials"
    yield stem + "_rows16", stem, ["LM_CE_VOCAB_TILE=16"]
    stem = "cross_entropy_linear_input_grad_streaming_fp16_native"
    yield stem, stem, []
    yield stem + "_compute", stem, ["LM_NATIVE_FP16_BACKWARD_COMPUTE=1"]
    for pairs in (64, 128, 224):
        yield stem + f"_reuse{pairs}", stem, [f"LM_WEIGHT_REUSE_PAIRS={pairs}"]
    for family in ("streaming", "tape"):
        stem = f"cross_entropy_linear_weight_grad_{family}_fp16_packed"
        for suffix, rows in (("", 8), ("_rows4", 4), ("_rows16", 16)):
            yield stem + suffix, stem, [f"LM_DW_VOCAB_ROWS_PER_GROUP={rows}"]
        if family == "streaming":
            for suffix, rows in (("", 8), ("_rows4", 4), ("_rows16", 16)):
                yield "cross_entropy_linear_weight_grad_streaming_fp16_native_compute" + suffix, stem, [
                    f"LM_DW_VOCAB_ROWS_PER_GROUP={rows}", "LM_NATIVE_FP16_BACKWARD_COMPUTE=1"]
    stem = "cross_entropy_linear_adjoints_tape_fp16_packed_fused"
    yield stem, stem, []
    for suffix, tile, workgroup in (
        ("", 64, 64), ("_tile128", 128, 64), ("_tile128_wg128", 128, 128),
        ("_tile256", 256, 64), ("_tile256_wg128", 256, 128), ("_tile256_wg256", 256, 256),
    ):
        yield stem + "_private_hidden" + suffix, stem, [
            "LM_CE_FUSED_PRIVATE_HIDDEN=1", f"LM_CE_FUSED_VOCAB_TILE={tile}",
            f"LM_CE_FUSED_WORKGROUP_SIZE={workgroup}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiler = shutil.which("glslc")
    validator = shutil.which("spirv-val")
    if compiler is None or validator is None:
        parser.error("glslc and spirv-val must be on PATH")
    shader_dir = Path(__file__).resolve().parents[1] / "shaders"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    with tempfile.TemporaryDirectory(prefix="hierarchos-ce-shaders-") as scratch:
        for artifact, source, defines in variants():
            rebuilt = Path(scratch) / f"{artifact}.spv"
            # Keep the existing CE modules' unoptimized arithmetic staging.
            flags = ["--target-env=vulkan1.1", *["-D" + value for value in defines]]
            source_path = shader_dir / f"{source}.comp"
            subprocess.run([compiler, *flags, str(source_path), "-o", str(rebuilt)], check=True)
            subprocess.run([validator, "--target-env", "vulkan1.1", str(rebuilt)], check=True)
            embedded = shader_dir / rebuilt.name
            if args.write:
                # Windows may briefly retain an antivirus/compiler file view.
                for attempt in range(5):
                    try:
                        replacement = embedded.with_suffix(".spv.tmp")
                        replacement.write_bytes(rebuilt.read_bytes())
                        replacement.replace(embedded)
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(0.1)
            rows.append({"artifact": embedded.name, "source": source_path.name,
                         "flags": flags, "source_sha256": digest(source_path),
                         "sha256": digest(embedded), "matches": digest(embedded) == digest(rebuilt)})
    report = {"compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
              "variants": rows, "pass": all(row["matches"] for row in rows)}
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

"""Independent V4.1 packed storage / block-scaled linear CPU oracle.

Hash-checks the released model, converter, and kernels. FP8/UE8M0 decoding and
BF16 casts use PyTorch. FP4 magnitudes use the converter's literal table, with
the E2M1 sign bit preserved for negative zero. GEMM uses independent CPU matrix
products per released K block, then the released scale/accumulation expression.
These component comparisons do not establish complete model logit parity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import validate_deepseek_v41_quantization as quant
import validate_deepseek_v41_engram as engram

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41_packed_weights.json"
MODEL_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
CONVERT_SHA256 = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"


class Fp32Projection(engram.LinearStub):
    """Exact BF16 promotion at the experimental graph's FP32 projection boundary."""
    def forward(self, x):
        return F.linear(x.float(), self.weight)


def bits(tensor):
    return tensor.contiguous().numpy().view(np.uint32).reshape(-1).tolist()


def unpack(data, rows, cols, fmt, fp4_table):
    raw = torch.tensor(data, dtype=torch.uint8)
    if fmt == "fp4_row32":
        codes = torch.stack((raw & 15, raw >> 4), -1).flatten().long()
        values = torch.tensor(fp4_table)[codes]
        values = torch.copysign(values.abs(), torch.where(codes & 8 != 0, -1.0, 1.0))
    else:
        values = raw.view(torch.float8_e4m3fn).float()
    return values.reshape(rows, cols)


def scale_matrix(scales, rows, cols, fmt):
    block = 128 if fmt == "fp8_block128" else 32
    sr = (rows + block - 1) // block if fmt.startswith("fp8_block") else rows
    values = torch.tensor(scales, dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(sr, cols // block)
    if fmt.startswith("fp8_block"):
        values = values.repeat_interleave(block, dim=0)[:rows]
    return values.repeat_interleave(block, dim=1)


def make_fixture(model_path, kernel_path, convert_path):
    for path, expected in [(model_path, MODEL_SHA256), (kernel_path, quant.KERNEL_SHA256), (convert_path, CONVERT_SHA256)]:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"pinned source hash mismatch: {path}, {actual}")
    tree = ast.parse(convert_path.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FP4_TABLE" for t in node.targets))
    fp4_table = ast.literal_eval(assignment.value.args[0])
    torch.set_num_threads(1)
    rng = np.random.default_rng(4141030)
    decodes, linears, engrams = [], [], []

    def decode_case(name, rows, cols, fmt, data, scales):
        values = unpack(data, rows, cols, fmt, fp4_table) * scale_matrix(scales, rows, cols, fmt)
        assert torch.isfinite(values).all() and torch.isfinite(values.bfloat16()).all()
        decodes.append(dict(name=name, rows=rows, cols=cols, format=fmt, data=list(map(int, data)), scales=list(map(int, scales)),
                            expected_fp32_bits=bits(values), expected_bf16_bits=bits(values.bfloat16().float())))

    finite = np.array([b for b in range(256) if b & 127 != 127], dtype=np.uint8)
    for fmt, rows, cols in [("fp8_block32", 35, 64), ("fp8_block128", 129, 128), ("fp4_row32", 17, 64), ("engram_fp8_row32", 5, 64)]:
        block = 128 if fmt == "fp8_block128" else 32
        sr = (rows + block - 1) // block if fmt.startswith("fp8_block") else rows
        data = np.resize(np.arange(256, dtype=np.uint8) if fmt == "fp4_row32" else finite,
                         rows * cols // (2 if fmt == "fp4_row32" else 1))
        scales = rng.integers(115, 132, sr * (cols // block), dtype=np.uint8)
        decode_case(f"{fmt}-all-codes-boundaries", rows, cols, fmt, data, scales)
    # Includes UE8M0 code 0 (2^-127), BF16 underflow and signed zero. Keep
    # finite weights at the largest exponent; overflow rejection is tested in Rust.
    decode_case("fp8-every-ue8m0-exponent", 255, 32, "engram_fp8_row32",
                np.resize([0, 128, 1, 129, 8, 136, 56, 184], 255 * 32), np.arange(255))
    decode_case("fp4-every-ue8m0-exponent", 255, 32, "fp4_row32",
                np.resize([0x10, 0x32, 0x98, 0xba], 255 * 16), np.arange(255))

    # Specs and seeds are fixed independently of native results. These are
    # component contracts, not changes to any existing model's weights/inputs.
    specs = [("fp8_block32", 32, 1, 35, 64), ("fp8_block32", 32, 5, 35, 96),
             ("fp8_block128", 128, 1, 7, 256), ("fp8_block128", 128, 5, 129, 256),
             ("fp4_row32", 32, 1, 17, 96), ("fp4_row32", 32, 5, 17, 160),
             ("fp4_row32", 128, 1, 17, 256), ("fp4_row32", 128, 5, 17, 256)]
    for fmt, act_block, m, n, k in specs:
        x = torch.from_numpy(rng.normal(0, 0.7, (m, k)).astype(np.float32)).bfloat16().float()
        if fmt == "fp4_row32":
            data = rng.integers(0, 256, n * k // 2, dtype=np.uint8)
        else:
            data = torch.from_numpy(rng.normal(0, 1, (n, k)).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).flatten().numpy()
        block = 128 if fmt == "fp8_block128" else 32
        sr = (n + block - 1) // block if fmt.startswith("fp8_block") else n
        scales = rng.integers(120, 125, sr * (k // block), dtype=np.uint8)
        w = unpack(data, n, k, fmt, fp4_table)
        sw = scale_matrix(scales, n, k, fmt)
        dequant, sa = quant.quantize_case(x.numpy(), act_block, "fp8", "ue8m0")
        a_scales = torch.from_numpy(sa).repeat_interleave(act_block, dim=1)
        aq = torch.from_numpy(dequant) / a_scales
        y = torch.zeros(m, n)
        for start in range(0, k, block):
            dot = F.linear(aq[:, start:start + block], w[:, start:start + block])
            y = y + (dot * a_scales[:, start:start + 1]) * sw[:, start].unsqueeze(0)
        linears.append(dict(name=f"{fmt}-act{act_block}-m{m}-n{n}-k{k}", rows=m, outputs=n, inner=k, format=fmt,
                            act_block=act_block, data=data.tolist(), scales=scales.tolist(), input=x.flatten().tolist(),
                            expected_fp32_bits=bits(y), expected_bf16_bits=bits(y.bfloat16().float())))
    namespace = engram.load_reference(model_path)
    embedding_class = next(n for n in ast.parse(model_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ParallelEngramEmbedding")
    exec(compile(ast.Module(body=[embedding_class], type_ignores=[]), str(model_path), "exec"), namespace)
    for rows, hc, dim, head_dim, table_rows, cols in [(3, 2, 8, 32, 5, 2), (17, 4, 32, 64, 11, 4)]:
        data = torch.from_numpy(rng.normal(0, 1, (table_rows, head_dim)).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8)
        scales = rng.integers(122, 129, (table_rows, head_dim // 32), dtype=np.uint8)
        embed = namespace["ParallelEngramEmbedding"].__new__(namespace["ParallelEngramEmbedding"])
        torch.nn.Module.__init__(embed)
        embed.weight = torch.nn.Parameter(data.view(torch.float8_e4m3fn), requires_grad=False)
        embed.scale = torch.nn.Parameter(torch.from_numpy(scales).view(torch.float8_e8m0fnu), requires_grad=False)
        embed.block_size, embed.vocab_start_idx, embed.vocab_end_idx = 32, 0, table_rows
        module = namespace["Engram"].__new__(namespace["Engram"])
        torch.nn.Module.__init__(module)
        module.dim, module.hc_mult, module.clamp_value, module.eps = dim, hc, 1e-6, 1e-20
        module.embed = embed
        wkv = torch.from_numpy(rng.normal(0, 0.1, ((hc + 1) * dim, cols * head_dim)).astype(np.float32))
        module.wkv = Fp32Projection(wkv)
        module.q_weight = torch.from_numpy(rng.normal(0, 0.3, (hc, dim)).astype(np.float32))
        module.k_weight = torch.from_numpy(rng.normal(0, 0.3, (hc, dim)).astype(np.float32))
        stream = torch.from_numpy(rng.normal(0, 0.4, (rows, hc, dim)).astype(np.float32))
        hashes = torch.from_numpy(rng.integers(-2, table_rows + 2, (rows, cols)))
        mask = torch.tensor([i % 3 != 1 for i in range(rows)])
        lookup = embed(hashes).float()
        expected = module(stream, hashes, mask)
        dense = unpack(data.flatten().tolist(), table_rows, head_dim, "engram_fp8_row32", fp4_table) * scale_matrix(scales.flatten().tolist(), table_rows, head_dim, "engram_fp8_row32")
        engrams.append(dict(name=f"packed-engram-rows{rows}-hc{hc}-hidden{dim}", rows=rows, hc=hc, hidden=dim, head_dim=head_dim,
                            table_rows=table_rows, cols=cols, data=data.flatten().tolist(), scales=scales.flatten().tolist(),
                            table=dense.bfloat16().float().flatten().tolist(), stream=stream.flatten().tolist(), hashes=hashes.flatten().tolist(), mask=mask.int().tolist(),
                            wkv=wkv.flatten().tolist(), q_weight=module.q_weight.flatten().tolist(), k_weight=module.k_weight.flatten().tolist(),
                            expected_lookup_bits=bits(lookup), expected=expected.flatten().tolist()))
    return dict(revision=quant.REVISION, model_sha256=MODEL_SHA256, kernel_sha256=quant.KERNEL_SHA256,
                converter_sha256=CONVERT_SHA256, torch_version=torch.__version__,
                torch_revision=torch.version.git_version, seed=4141030, decodes=decodes, linears=linears, engrams=engrams)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--kernel-reference-path", type=Path, required=True)
    parser.add_argument("--convert-reference-path", type=Path, required=True)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture(args.reference_path, args.kernel_reference_path, args.convert_reference_path)
    if args.write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != fixture:
        raise AssertionError("packed fixture differs from the independent CPU reference")
    print(f"PASS packed CPU oracle: {len(fixture['decodes'])} decode cases, {len(fixture['linears'])} linear cases, {len(fixture['engrams'])} released Engram cases")


if __name__ == "__main__":
    main()

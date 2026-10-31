"""Complete hybrid text-model logits: FP32 backbone + real packed MoE modules.

This preserves released Expert/MoE bodies and quantized GEMM BF16 boundaries.
It qualifies the CPU-reference hybrid; production BF16 execution remains separate.
Original FP32 acceptance fixtures and their strict comparator stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import validate_deepseek_v41_model as model
import validate_deepseek_v41_mixed as mixed
import validate_deepseek_v41_quantization as quant

FIXTURE = model.FIXTURE.parent / "deepseek_v41_mixed_model.json"
SEEDS = [4141041,4141042]


def quantize_weight(tensor, dtype):
    values = tensor.detach().float().numpy()
    rows, cols = values.shape
    assert cols % 32 == 0
    if dtype == "fp4_row32":
        dequant, scales = quant.quantize_case(values,32,"fp4","ue8m0")
        normalized = dequant / np.repeat(scales,32,axis=1)
        levels = [0.,.5,1.,1.5,2.,3.,4.,6.]
        codes = np.array([levels.index(abs(float(v))) + (8 if float(v)<0 else 0) for v in normalized.flat], dtype=np.uint8)
        data = (codes[0::2] | (codes[1::2] << 4)).tolist()
    else:
        padded = torch.zeros((rows+31)//32*32,cols)
        padded[:rows] = torch.from_numpy(values.copy())
        blocks = padded.reshape((rows+31)//32,32,cols//32,32).permute(0,2,1,3)
        amax = blocks.abs().amax((-1,-2)).clamp_min(1e-4)
        scales = np.array([[quant.fast_pow2(quant.fast_log2_ceil(np.float32(v)*np.float32(1/448))) for v in row] for row in amax.numpy()],dtype=np.float32)
        expanded = torch.from_numpy(scales).repeat_interleave(32,0).repeat_interleave(32,1)
        data = (torch.from_numpy(values.copy())/expanded[:rows]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist()
    scale_bytes = torch.from_numpy(scales).to(torch.float8_e8m0fnu).view(torch.uint8).flatten().tolist()
    return dict(rows=rows,cols=cols,dtype=dtype,data=data,scales=scale_bytes)


def make_case(path,kernel_path,seed):
    ns = mixed.source_namespace(path)
    packed_tensors = {}
    def transform(net):
        for layer in net.layers:
            moe = layer.ffn
            for index, expert in enumerate(moe.experts):
                prefix = f"layers.{layer.layer_id}.ffn.experts.{index}"
                spec = {name:quantize_weight(getattr(expert,name).weight,"fp4_row32") for name in ("w1","w2","w3")}
                moe.experts[index] = mixed.expert(ns,spec,expert.swiglu_limit)
                for name,w in spec.items(): packed_tensors[f"{prefix}.{name}.weight"] = w
            expert = moe.shared_experts
            spec = {name:quantize_weight(getattr(expert,name).weight,"fp8_block32") for name in ("w1","w2","w3")}
            moe.shared_experts = mixed.expert(ns,spec,expert.swiglu_limit)
            for name,w in spec.items(): packed_tensors[f"layers.{layer.layer_id}.ffn.shared_experts.{name}.weight"] = w
    f = model.make_fixture(path,kernel_path,seed=seed,args_overrides={"moe_inter_dim":32},model_transform=transform)
    f["config"]["text_config"]["moe_intermediate_size"] = 32
    f["precision"] = "fp32_backbone_with_packed_fp4_routed_fp8_shared_experts_and_bf16_gemm_outputs_cpu_mirror"
    f["packed_tensors"] = packed_tensors
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-path",type=Path,required=True)
    p.add_argument("--kernel-reference-path",type=Path,required=True)
    p.add_argument("--write-fixture",action="store_true")
    p.add_argument("--native-output",type=Path)
    args = p.parse_args()
    f = dict(cases=[make_case(args.reference_path,args.kernel_reference_path,seed) for seed in SEEDS])
    if args.write_fixture: FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text()) != f: raise AssertionError("mixed-model fixture differs from independent source")
    print(f"PASS hybrid model oracle: seeds {SEEDS}, {sum(len(c['steps']) for c in f['cases'])} full-logit steps")
    if args.native_output:
        native = json.loads(args.native_output.read_text())
        assert len(native["cases"]) == len(f["cases"])
        binding = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for expected,actual in zip(f["cases"],native["cases"]):
            assert expected["seed"] == actual["seed"]
            print(f"HYBRID seed {expected['seed']}")
            model.compare_native(expected,actual,fixture_sha256=binding)


if __name__ == "__main__": main()

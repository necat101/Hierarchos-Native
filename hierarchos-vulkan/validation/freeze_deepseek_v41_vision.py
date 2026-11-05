"""Freeze training of the released V4.1 ViT/aligner, without native GPU input.

The released processor produces BF16 patches. This FP32 qualification promotes
those exact values to FP32 at the input boundary; all released module bodies
then execute unchanged. Existing text and image fixtures are never overwritten.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
REFERENCE = Path(__file__).parent / "reference/deepseek_v41_vl" / REVISION
HASHES = {
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
}


def load_references():
    modules = {}
    for name, expected in HASHES.items():
        path = REFERENCE / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Released reference hash changed: {name}")
        if name == "model.py":
            continue
        key = "v41_released_" + path.stem
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        modules[path.stem] = module
    return modules["vision"], modules["image_processor"]


def tensor(x):
    x = x.detach().float().contiguous()
    return {"shape": list(x.shape), "values": x.flatten().tolist()}


def arguments():
    return dict(vision_dim=16, vision_n_heads=2, vision_inter_dim=32,
                vision_n_layers=2, vision_patch_size=2, vision_rope_theta=10000.0,
                vision_downsample_ratio=2, vision_max_n_token=32,
                vision_min_pixels=0, vision_max_wh_ratio=None, dim=32)


def image_input(processor, args, width, height):
    rgb = ((np.arange(height * width * 3, dtype=np.uint32) * 37 + 19) % 256).astype(np.uint8)
    rgb = rgb.reshape(height, width, 3)
    encoded = io.BytesIO()
    Image.fromarray(rgb).save(encoded, format="PNG")
    patches, h, w, lh, lw = processor.load_image({"data": encoded.getvalue()}, args)
    assert patches.dtype == torch.bfloat16
    return patches.float(), dict(width=width, height=height, rgb=rgb.flatten().tolist(),
                                png_sha256=hashlib.sha256(encoded.getvalue()).hexdigest(),
                                n_vit_h=h, n_vit_w=w, n_llm_h=lh, n_llm_w=lw,
                                token_types=processor.image_token_types(lh, lw).tolist())


def case(vision, processor, width, height):
    torch.manual_seed(0x563431 + width * 100 + height)
    args = SimpleNamespace(**arguments())
    net = nn.ModuleDict({"vision": vision.ViT(args), "aligner": vision.Aligner(args)})
    with torch.no_grad():
        for name, p in net.named_parameters():
            if ".norm" in name:
                p.copy_(1.0 + torch.randn_like(p) * 0.02)
            else:
                p.copy_(torch.randn_like(p) * (0.02 if name.endswith("bias") else 0.08))
    patches, image = image_input(processor, args, width, height)
    patches.requires_grad_()
    initial = {n: tensor(p) for n, p in net.named_parameters()}
    h, w = image["n_vit_h"], image["n_vit_w"]
    upstream = torch.randn(image["n_llm_h"] * image["n_llm_w"], args.dim) * 0.03
    hyper = dict(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=0.001, weight_decay=0.0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=hyper["learning_rate"],
                                 betas=(hyper["beta1"], hyper["beta2"]), eps=hyper["eps"],
                                 weight_decay=hyper["weight_decay"], foreach=False, fused=False)
    steps = []
    for step in range(3):
        trace = {}
        handles = []

        def save(name, value):
            if isinstance(value, torch.Tensor):
                trace[name] = value
                if value.requires_grad:
                    value.retain_grad()

        if step == 0:
            for name, module in net.named_modules():
                if name:
                    handles.append(module.register_forward_hook(
                        lambda m, ins, out, name=name: save(name + ".output", out)))
            original_sdpa = F.scaled_dot_product_attention

            def record_sdpa(q, k, v, *a, **kw):
                name = "sdpa." + str(sum(key.endswith(".q") for key in trace))
                for label, x in [("q", q), ("k", k), ("v", v)]:
                    save(name + "." + label, x)
                result = original_sdpa(q, k, v, *a, **kw)
                save(name + ".output", result)
                return result

            F.scaled_dot_product_attention = record_sdpa
        try:
            features = net["vision"](patches, h, w)
            projected = net["aligner"](features, h, w)
        finally:
            if step == 0:
                F.scaled_dot_product_attention = original_sdpa
            for handle in handles:
                handle.remove()
        record = dict(features=tensor(features), projected=tensor(projected))
        if step < 2:
            optimizer.zero_grad(set_to_none=True)
            patches.grad = None
            projected.backward(upstream)
            record["patch_gradient"] = tensor(patches.grad)
            record["gradients"] = {n: tensor(p.grad) for n, p in net.named_parameters()}
            assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in net.parameters())
            if step == 0:
                record["trace"] = {n: dict(value=tensor(x), gradient=None if x.grad is None else tensor(x.grad))
                                   for n, x in trace.items()}
            optimizer.step()
            record["updated_parameters"] = {n: tensor(p) for n, p in net.named_parameters()}
        steps.append(record)
    assert all(initial[n]["values"] != tensor(p)["values"] for n, p in net.named_parameters())
    return dict(name=f"native_vit_{width}x{height}", config=arguments(), image=image,
                patches=tensor(patches), upstream=tensor(upstream), parameters=initial,
                hyper=hyper, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("The frozen CPU oracle requires AVX2")
    vision, processor = load_references()
    payload = dict(schema=1, tolerance=2e-7, rtol=0, comparison="strict_less_than",
                   model="deepseek-ai/DeepSeek-V4.1-Flash-1017", revision=REVISION,
                   source_sha256=HASHES, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch=torch.__version__, torch_revision=torch.version.git_version,
                   cpu=torch.backends.cpu.get_cpu_capability(), threads=1,
                   contract="Unchanged released processor/ViT/aligner; BF16 processor patches promoted to FP32; two AdamW updates; independent output VJP. Standalone vision evidence, not full multimodal qualification.",
                   cases=[case(vision, processor, 4, 4), case(vision, processor, 6, 4)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(payload, out, indent=1)
        out.write("\n")
    print("Frozen", len(payload["cases"]), "native V4.1 vision training cases", flush=True)


if __name__ == "__main__":
    main()

"""Independent complete CPU model autograd/AdamW oracle.

The released model/block/attention/Engram/MoE bodies execute unchanged. Three
inference-buffer updates are lifted to functional copies so cached gradients
stay live, and quantizer rounding uses the established FP32 STE. No native
output or gradient supplies expected values.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import validate_deepseek_v41_model as model
from validate_deepseek_v41_block_grad import bits
from validate_deepseek_v41_quantization import quantize_case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "fixtures/deepseek_v41_full_training.json"
SOURCE = ROOT / ".tmp-v41-ref/model.py"
KERNEL = ROOT / "target/deepseek-v41-reference/inference_kernel.py"
BASE = HERE / "fixtures/deepseek_v41_model.json"


class Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def install_functional(ns):
    def activation(x, block, mode, scale):
        original = x.clone()
        shape = x.shape
        values, _ = quantize_case(x.detach().float().reshape(-1, shape[-1]).numpy(), block, mode, scale)
        x.copy_(Boundary.apply(original, torch.from_numpy(values).reshape(shape)))
        return x

    ns["act_quant"] = lambda x, block, fmt, dtype, dequant: activation(x, block, "fp8", "ue8m0")
    ns["fp4_act_quant"] = lambda x, block, dequant, scale_dtype=None: activation(x, block, "fp4", "e4m3" if scale_dtype == torch.float8_e4m3fn else "ue8m0")

    def rope(x, freqs, inverse=False):
        original = x.clone()
        freq = freqs.conj() if inverse else freqs
        # Adjacent-pair real arithmetic is the kernel mirror used by the
        # already pinned rotary adjoint; no complex multiply reassociation.
        shape = [1] * original.ndim
        shape[1], shape[-1] = original.shape[1], original.shape[-1] // 2
        c, s = freq.real.reshape(shape), freq.imag.reshape(shape)
        re, im = original[..., ::2], original[..., 1::2]
        x.copy_(torch.stack((re*c-im*s, re*s+im*c), -1).flatten(-2))
        return x
    ns["apply_rotary_emb"] = rope

    def compressor(self, x, position):
        batch, seq, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x))
        kv, score = self.wkv(x.float()), self.wgate(x.float())
        state_k = self.kv_state.clone()
        state_s = self.score_state.clone()
        if position == 0:
            tail, groups = seq % ratio, seq // ratio
            state_k = torch.zeros_like(state_k)
            state_s = torch.full_like(state_s, -torch.inf)
            if tail:
                state_k[:batch, :tail] = kv[:, groups*ratio:]
                state_s[:batch, :tail] = score[:, groups*ratio:]
            self.kv_state, self.score_state = state_k, state_s
            if not groups:
                return None
            values = kv[:, :groups*ratio].unflatten(1, (groups, ratio))
            weights = score[:, :groups*ratio].unflatten(1, (groups, ratio)).softmax(2)
            pooled = (values*weights).sum(2)
        else:
            state_k[:batch, position % ratio] = kv.squeeze(1)
            state_s[:batch, position % ratio] = score.squeeze(1)
            self.kv_state, self.score_state = state_k, state_s
            if (position+1) % ratio:
                return None
            pooled = (state_k[:batch]*state_s[:batch].softmax(1)).sum(1, keepdim=True)
        return self.norm(pooled)
    ns["Compressor"].forward = compressor

    def window(self, x, freqs, position):
        batch, seq, _ = x.shape
        kv = self.kv_norm(self.wkv(x))
        rope(kv[..., -self.rope_head_dim:], freqs)
        activation(kv, 32, "fp8", "ue8m0")
        cache = self.window_kv_cache.clone()
        for row in range(seq):
            if position > 0 or row >= seq-self.window_size:
                cache[:batch, (position+row) % self.window_size] = kv[:, row]
        self.window_kv_cache = cache
        return (kv if position == 0 else cache[:batch], ns["get_window_topk_idxs"](self.window_size,batch,seq,position))
    ns["Attention"]._window_kv = window

    def compressed(self, x, qr, position, offset):
        batch, seq, _ = x.shape
        ratio = self.compress_ratio
        count = (position+seq)//ratio
        latent = self.compressor(x,position) if self.is_kv_source else None
        if self.is_kv_source:
            ns["shared_attn"].compress_kv = self.compress_kv_cache
        # Index selection is a fixed discrete edge. Execute its unchanged
        # forward with the actual adapter values and shared key/candidate state.
        with torch.no_grad():
            indices = self._compress_topk_idxs(x,qr,latent,position,offset,count)
        if latent is not None:
            freqs = self.freqs_cis[:seq-seq%ratio:ratio] if position == 0 else self.freqs_cis[position+1-ratio].unsqueeze(0)
            rope(latent[..., -self.rope_head_dim:],freqs)
            activation(latent,16,"fp4","e4m3")
            cache = self.compress_kv_cache.clone()
            cache[:batch,position//ratio:position//ratio+latent.shape[1]] = latent
            self.compress_kv_cache = cache
            ns["shared_attn"].compress_kv = cache
        return ns["shared_attn"].compress_kv[:batch,:count], indices
    ns["Attention"]._compress_kv = compressed


class Lora(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.a, self.b, self.scale = base, a, b, scale

    def forward(self, x):
        return self.base(x) + self.scale*F.linear(F.linear(x,self.a),self.b)


def supported(target):
    if target.endswith((".attn.wq_a", ".attn.wq_b", ".attn.wkv", ".attn.wo_b", ".engram.wkv")):
        return True
    if ".attn.indexer." in target and target.rsplit(".",1)[1] in {"wk","wq_b","weights_proj"}:
        return True
    return (".ffn.experts." in target or ".ffn.shared_experts." in target) and target.rsplit(".",1)[1] in {"w1","w2","w3"}


def make_parameters(base, rslora=False):
    state = 1793
    specs, params = {}, {}
    rank, alpha = 2, 2.0
    for name,tensor in sorted(base["tensors"].items()):
        if not name.endswith(".weight") or not supported(name[:-7]):
            continue
        target = name[:-7]
        rows, cols = tensor["shape"]
        bound = np.float32(1/math.sqrt(cols))
        a = []
        for _ in range(rank*cols):
            state = (state*6364136223846793005+1442695040888963407) & ((1<<64)-1)
            unit = np.float32(np.float32(state>>40)/np.float32(16777216))
            a.append(np.float32(np.float32(np.float32(unit*np.float32(2))-np.float32(1))*bound))
        a = nn.Parameter(torch.tensor(a).reshape(rank,cols))
        b = nn.Parameter(torch.zeros(rows,rank))
        scale = float(np.float32(alpha/(math.sqrt(rank) if rslora else rank)))
        specs[target] = dict(rank=rank,alpha=alpha,scale=scale,rows=rows,cols=cols,a_bits=bits(a),b_bits=bits(b))
        params[target] = (a,b)
    return specs, params


def build(base, params, specs, *, args_overrides=None):
    ns = model.load_reference(SOURCE,KERNEL,tokens=base["tokens"])
    install_functional(ns)
    args = ns["ModelArgs"](max_batch_size=2,max_seq_len=16,temperature=0,dtype="bf16",expert_dtype=None,
        vocab_size=16,dim=32,moe_inter_dim=12,n_layers=6,n_mtp_layers=0,n_heads=2,
        n_routed_experts=3,n_activated_experts=2,route_scale=1.5,swiglu_limit=1.5,
        q_lora_rank=8,head_dim=32,rope_head_dim=16,o_groups=1,o_lora_rank=16,window_size=4,
        compress_ratios=(0,2,2,1,1,1),kv_source_layers=(1,3),index_source_layers=(1,3,5),
        compress_rope_theta=160000,original_seq_len=0,index_n_heads=2,index_head_dim=32,index_topk=2,
        candidate_source_layer=3,candidate_topk_blocks=2,candidate_block_size=2,hc_mult=2,
        engram_layer_ids=(0,4),engram_num_embeddings=(128,128),engram_max_ngram_size=3,
        engram_vocab_size=5,engram_n_heads=2,engram_head_dim=4,engram_compressed_vocab_size=8)
    for key,value in (args_overrides or {}).items():
        setattr(args,key,value)
    net = ns["Transformer"](args)
    state = {name:torch.tensor(t["values"],dtype=torch.float32).reshape(t["shape"]) for name,t in base["tensors"].items()}
    net.load_state_dict(state,strict=True)
    net.requires_grad_(False)
    for target,(a,b) in params.items():
        path,leaf = target.rsplit(".",1)
        owner = net.get_submodule(path)
        setattr(owner,leaf,Lora(getattr(owner,leaf),a,b,specs[target]["scale"]))
    return ns,net


def case(base, rslora):
    specs, params = make_parameters(base,rslora)
    hyper = dict(learning_rate=1e-3,beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0.01)
    optimizer = torch.optim.AdamW([p for pair in params.values() for p in pair],lr=hyper["learning_rate"],
        betas=(hyper["beta1"],hyper["beta2"]),eps=hyper["eps"],weight_decay=hyper["weight_decay"],foreach=False)
    batches = [dict(token_ids=s["ids"],targets=[-100 if i == 0 else (v+1)%16 for i,v in enumerate(s["ids"])],
                    seq_len=s["seq"],engram_mask=s["mask"],position=s["pos"]) for s in base["steps"][:4]]
    steps = []
    for step in range(3):
        ns,net = build(base,params,specs)
        captured = []
        hook = net.head.register_forward_pre_hook(lambda _,inputs:captured.append(inputs[0]))
        logits = []
        for batch in batches:
            ids = torch.tensor(batch["token_ids"]).reshape(2,batch["seq_len"])
            types = torch.tensor([-1 if v else 0 for v in batch["engram_mask"]]).reshape_as(ids)
            ns["Transformer"].forward.__wrapped__(net,ids,batch["position"],token_types=types)
            logits.append(F.linear(captured[-1],net.head.weight).reshape(-1,16))
        hook.remove()
        targets = torch.tensor([v for batch in batches for v in batch["targets"]])
        loss = F.cross_entropy(torch.cat(logits),targets,ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {target:None if a.grad is None else dict(a_bits=bits(a.grad),b_bits=bits(b.grad)) for target,(a,b) in params.items()}
        assert any(v is not None for v in gradients.values())
        assert all(gradients[t] is None for t in gradients if ".indexer." in t)
        optimizer.step()
        states = {}
        for target,(a,b) in params.items():
            value = dict(a_bits=bits(a),b_bits=bits(b))
            for label,p in [("a",a),("b",b)]:
                state = optimizer.state.get(p,{})
                value["step_"+label] = int(state["step"].item()) if state else 0
                value["m_"+label] = bits(state["exp_avg"]) if state else [0]*p.numel()
                value["v_"+label] = bits(state["exp_avg_sq"]) if state else [0]*p.numel()
            states[target] = value
        steps.append(dict(loss=loss.item(),logits_bits=[bits(x) for x in logits],gradients=gradients,states=states))
        print("CPU full training",rslora,step,"loss",loss.item(),"active",sum(g is not None for g in gradients.values()),flush=True)
    return dict(name="rslora" if rslora else "lora",rslora=rslora,seed=1793,modules=specs,hyper=hyper,batches=batches,steps=steps)


def main():
    torch.set_num_threads(1)
    base = json.loads(BASE.read_text())
    payload = dict(tolerance=2e-7,rtol=0,comparison="strict_less_than",base_fixture_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),torch_version=torch.__version__,torch_git_version=torch.version.git_version,
        contract="Released model bodies, functional inference-cache writes, FP32 straight-through quantizer boundaries, fixed discrete index selection",
        cases=[case(base,False),case(base,True)])
    if FIXTURE.exists() and json.loads(FIXTURE.read_text()) != payload:
        raise SystemExit("full training fixture changed; refusing to overwrite frozen CPU evidence")
    FIXTURE.write_text(json.dumps(payload,indent=2)+"\n")
    print("PASS CPU full model training fixture",hashlib.sha256(FIXTURE.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

"""Full BF16/packed CPU training mirror with explicit FP32 STE cast boundaries.

Forward is checked against the independently frozen released BF16 model before
any training evidence is accepted. Packed GEMMs use the existing adjoint
contract; the released Block, Expert, Engram, Gate and Transformer bodies run.
"""
import copy
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
import validate_deepseek_v41_full_training as full
from validate_deepseek_v41_packed_grad import packed_forward

HERE=Path(__file__).resolve().parent
BASE=HERE/'fixtures/deepseek_v41_bf16_model.json'
FIXTURE=HERE/'fixtures/deepseek_v41_full_bf16_training_v3.json'
original_build=full.build

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.bfloat16().float()
    @staticmethod
    def backward(ctx,g):return g
bf16=Round.apply

class Projection(nn.Module):
    def __init__(self,weight,spec,a=None,b=None,scale=0):
        super().__init__();self.weight=weight;self.spec=spec;self.a=a;self.b=b;self.scale=scale
    def forward(self,x):
        # Every adapted projection sees its enclosing stream tensor. In
        # particular Expert.forward casts SwiGLU to BF16 BEFORE calling w2;
        # the adapter must consume that same rounded input as its frozen base.
        shape=x.shape;y=bf16(x.reshape(-1,shape[-1]))
        if self.spec is not None:
            out=packed_forward(y,self.spec,True,self.a,self.b,self.scale)
        else:
            out=bf16(F.linear(bf16(y),self.weight))
            if self.a is not None:out=bf16(out+self.scale*F.linear(F.linear(y,self.a),self.b))
        return out.reshape(*shape[:-1],-1)

def constructor_base(original):
    base=copy.deepcopy(original)
    for name,spec in base['packed_tensors'].items():
        base['tensors'][name]=dict(shape=[spec['rows'],spec['cols']],values=[0.0]*(spec['rows']*spec['cols']))
    return base

def build(base,params,specs):
    text=base['config']['text_config']
    ns,net=original_build(base,params,specs,args_overrides=dict(q_lora_rank=text['q_lora_rank'],o_lora_rank=text['o_lora_rank'],
        moe_inter_dim=text['moe_intermediate_size'],hc_mult=text['hc_mult'],engram_head_dim=text['engram_head_dim'],
        n_routed_experts=text['n_routed_experts'],n_activated_experts=text['num_experts_per_tok']))
    for name,tensor in base['tensors'].items():
        if not name.endswith('.weight'):continue
        target=name[:-7];spec=base['packed_tensors'].get(name)
        if target in {'embed','head'}:continue
        if target.endswith('.engram.embed'):
            if spec is not None:
                data=torch.tensor(spec['data'],dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(spec['rows'],spec['cols'])
                scale=torch.tensor(spec['scales'],dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(spec['rows'],-1)
                net.get_submodule(target).weight.data.copy_(data*scale.repeat_interleave(32,-1))
            continue
        if len(tensor['shape'])!=2 or (spec is None and tensor.get('dtype')!='bf16'):continue
        path,leaf=target.rsplit('.',1) if '.' in target else ('',target)
        owner=net.get_submodule(path) if path else net
        current=getattr(owner,leaf)
        if not isinstance(current,(full.Lora,full.model.Linear)):continue
        weight=(current.base if isinstance(current,full.Lora) else current).weight
        a,b=params.get(target,(None,None))
        setattr(owner,leaf,Projection(weight,spec,a,b,specs[target]['scale'] if a is not None else 0))
    # Pool projections/state remain FP32. The released Compressor rounds the
    # pooled value to the stream before RMSNorm, then returns that stream dtype.
    for name,module in net.named_modules():
        if isinstance(module,ns['MoE']):module.register_forward_hook(lambda m,a,y:bf16(y))
        if isinstance(module,ns['RMSNorm']):
            pool=name.endswith('.compressor.norm') and net.get_submodule(name.rsplit('.',1)[0]).compress_ratio>1
            if pool:module.register_forward_pre_hook(lambda m,a:(bf16(a[0]),))
            module.register_forward_hook(lambda m,a,y:bf16(y))
        if name.endswith('.indexer.k_norm'):module.register_forward_hook(lambda m,a,y:bf16(y))
    for method in ['hc_pre','hc_post']:
        old=getattr(ns['Block'],method)
        setattr(ns['Block'],method,lambda self,*args,old=old:bf16(old(self,*args)))
    engram=ns['Engram'].forward
    ns['Engram'].forward=lambda self,*a,**kw:bf16(engram(self,*a,**kw))
    rope=ns['apply_rotary_emb']
    quant=ns['act_quant'];fp4=ns['fp4_act_quant']
    def rotate(x,freqs,inverse=False,half=True):
        rope(x,freqs,inverse)
        if half:x.copy_(bf16(x.clone()))
        return x
    def window(self,x,freqs,pos):
        batch,seq,_=x.shape;kv=self.kv_norm(self.wkv(x))
        rotate(kv[...,-self.rope_head_dim:],freqs)
        quant(kv,32,'ue8m0',torch.float8_e8m0fnu,True)
        kv=bf16(kv);cache=self.window_kv_cache.clone()
        for row in range(seq):
            if pos>0 or row>=seq-self.window_size:cache[:batch,(pos+row)%self.window_size]=kv[:,row]
        self.window_kv_cache=cache
        return (kv if pos==0 else cache[:batch],ns['get_window_topk_idxs'](self.window_size,batch,seq,pos))
    def compressed(self,x,qr,pos,offset):
        batch,seq,_=x.shape;ratio=self.compress_ratio;count=(pos+seq)//ratio
        latent=self.compressor(x,pos) if self.is_kv_source else None
        if self.is_kv_source:ns['shared_attn'].compress_kv=self.compress_kv_cache
        with torch.no_grad():indices=self._compress_topk_idxs(x,qr,latent,pos,offset,count)
        if latent is not None:
            freqs=self.freqs_cis[:seq-seq%ratio:ratio] if pos==0 else self.freqs_cis[pos+1-ratio].unsqueeze(0)
            rotate(latent[...,-self.rope_head_dim:],freqs)
            fp4(latent,16,True,scale_dtype=torch.float8_e4m3fn)
            latent=bf16(latent);cache=self.compress_kv_cache.clone()
            cache[:batch,pos//ratio:pos//ratio+latent.shape[1]]=latent
            self.compress_kv_cache=cache;ns['shared_attn'].compress_kv=cache
        return ns['shared_attn'].compress_kv[:batch,:count],indices
    def attention(self,x,pos):
        batch,seq,_=x.shape;freqs=self.freqs_cis[pos:pos+seq];rd=self.rope_head_dim
        qr=self.q_norm(self.wq_a(x));q=self.wq_b(qr).unflatten(-1,(self.n_local_heads,self.head_dim))
        rotate(q[...,-rd:],freqs)
        kv,idx=window(self,x,freqs,pos)
        if self.compress_ratio:
            ck,ci=compressed(self,x,qr,pos,kv.size(1));kv=torch.cat([kv,ck],1);idx=torch.cat([idx,ci],-1)
        out=bf16(ns['sparse_attn'](q,kv,self.attn_sink,idx,self.softmax_scale))
        rotate(out[...,-rd:],freqs,True)
        out=out.view(batch,seq,self.n_local_groups,-1);quant(out,32,'ue8m0',torch.float8_e8m0fnu,True)
        out=bf16(out);weight=self.wo_a.weight.view(self.n_local_groups,self.o_lora_rank,-1)
        out=bf16(torch.einsum('bsgd,grd->bsgr',out,weight))
        return self.wo_b(out.flatten(2))
    ns['apply_rotary_emb']=rotate
    ns['Attention'].forward=attention
    install_indexer(ns)
    return ns,net


def install_indexer(ns):
    """Released `Indexer.forward` with the released dtype flow's BF16 boundaries.

    The released indexer runs on bf16 tensors, so `torch.einsum` returns bf16,
    the head-weighted product is bf16, and the head sum is a bf16 reduction.
    This mirror keeps that arithmetic: the same three roundings on the same
    operands, with the released key-cache write, candidate pre-filter and top-k
    mapping unchanged. Skipping them leaves a wider fp32 score, which only
    matters when two positions round together in bf16: the discrete ranking, not
    the arithmetic, is what the boundary decides.
    """
    def indexer(self, x, qr, latent, start_pos, offset):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        if self.owns_k and latent is not None:
            freqs = (self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0
                     else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
            k = self.k_norm(self.wk(latent))
            ns['apply_rotary_emb'](k[..., -rd:], freqs)
            ns['fp4_act_quant'](k, 32, True)
            self.k_cache[:bsz, start_pos // ratio:start_pos // ratio + k.size(1)] = k
        if self.owns_k:
            ns['shared_attn'].index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        ns['apply_rotary_emb'](q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        ns['fp4_act_quant'](q, 32, True)
        index_k = ns['shared_attn'].index_k[:bsz, :end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # bf16 einsum, bf16 product, bf16 head sum: the released tensor dtypes.
        scores = torch.einsum('bshd,btd->bsht', q.bfloat16(), index_k.bfloat16())
        scores = (scores.relu_() * weights.bfloat16().unsqueeze(-1)).sum(dim=2)
        index_score = bf16(scores.float())
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        if self.is_candidate_source:
            ns['shared_attn'].candidates = ns['select_candidate_blocks'](
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~ns['shared_attn'].candidates, -torch.inf)
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()
    ns['Indexer'].forward = indexer

def main():
    torch.set_num_threads(1)
    original=json.loads(BASE.read_text())['cases'][0]
    base=constructor_base(original)
    specs,params=full.make_parameters(base,False)
    ns,net=build(base,params,specs)
    captured=[];net.head.register_forward_pre_hook(lambda m,a:captured.append(a[0]))
    for step in original['steps'][:4]:
        ids=torch.tensor(step['ids']).reshape(2,step['seq'])
        types=torch.tensor([-1 if v else 0 for v in step['mask']]).reshape_as(ids)
        ns['Transformer'].forward.__wrapped__(net,ids,step['pos'],token_types=types)
        logits=F.linear(captured[-1],net.head.weight).flatten()
        expected=torch.tensor(step['logits']);drift=(logits-expected).abs().max().item()
        print('BF16 training mirror initial forward',step['name'],drift,flush=True)
        assert drift<2e-7
    full.build=build
    cases=[]
    for rslora in [False,True]:
        c=full.case(base,rslora);c['base_fixture']=BASE.name;c['base_seed']=original['seed'];cases.append(c)
    payload=dict(tolerance=2e-7,rtol=0,comparison='strict_less_than',base_fixture_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(full.SOURCE.read_bytes()).hexdigest(),torch_version=torch.__version__,torch_git_version=torch.version.git_version,
        contract='Full BF16/FP4/FP8 model with established FP32 STE cast and quantizer boundaries, including the indexer bf16 einsum/product/head-sum boundaries; unchanged released block bodies',cases=cases)
    if FIXTURE.exists() and json.loads(FIXTURE.read_text())!=payload:raise SystemExit('BF16 CPU training fixture changed; refusing overwrite')
    FIXTURE.write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS full BF16 CPU training fixture',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())

if __name__=='__main__':main()

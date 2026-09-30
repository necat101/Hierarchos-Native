//! Falcon H1 parallel Mamba2 branch. No Python or host tensor math at runtime.
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falcon_h1_rms_sqrt_div_rounding_matches_ieee() -> Result<()> {
        let device = VulkanDevice::new()?;
        let kernel = vulkan::ComputeKernel::new(&device, RMS_NORM_FORWARD_SPV, 5,
            std::mem::size_of::<RmsNormForwardPush>() as u32)?;
        // Cover many normal significands, exponent transitions, exact powers
        // of two and both sides of them. Host math is the test oracle only.
        let mut values=Vec::new();
        let mut state=0x31415926u32;
        for exponent in 80u32..176 {
            for mantissa in [0u32, 1, 0x7ffffe, 0x7fffff] {
                values.push(f32::from_bits((exponent<<23)|mantissa));
            }
            for _ in 0..64 {
                state=state.wrapping_mul(1664525).wrapping_add(1013904223);
                values.push(f32::from_bits((exponent<<23)|(state&0x7fffff)));
            }
        }
        let rows=values.len()/3;
        let input=GpuBuffer::from_f32(&device,&values)?;
        let weight=GpuBuffer::from_f32(&device,&[1.0;3])?;
        let output=GpuBuffer::zeros_f32(&device,rows*3)?;
        let rstd=GpuBuffer::zeros_f32(&device,rows)?;
        let mean=GpuBuffer::zeros_f32(&device,rows)?;
        let push=RmsNormForwardPush { rows:rows as u32,dim:3,parameter_groups:1,
            eps:0.0,weight_offset:0.0,pow_rstd:0,unfused_products:1,extra_rsqrt_refinement:1 };
        let mut cmd=vulkan::ComputeBatch::new(&device)?;
        kernel.record_dispatch(&mut cmd,&[&input,&weight,&output,&rstd,&mean],
            bytemuck::bytes_of(&push),[div_ceil_u32(rows,64),1,1])?;
        cmd.submit()?;
        let means=mean.read_f32(rows)?;
        let actual=rstd.read_f32(rows)?;
        for (row,(&x,&y)) in means.iter().zip(&actual).enumerate() {
            let expected=1.0f32/x.sqrt();
            assert_eq!(y.to_bits(),expected.to_bits(),"row {row}, variance {x:e}");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    mode: u32, batch: u32, seq: u32, inner: u32,
    heads: u32, groups: u32, state_dim: u32, kernel: u32,
    chunk: u32, norm: u32, norm_before: u32, cached: u32,
    eps: f32, dt_min: f32, dt_max: f32, mz: f32,
    mx: f32, mb: f32, mc: f32, mdt: f32,
    conv_bias: u32, _pad: [u32;3],
}

pub(super) struct Mixer {
    push: Push,
    kernel: vulkan::ComputeKernel,
    in_proj: VulkanLinear,
    out_proj: VulkanLinear,
    params: TrainableParameter,
    train_params: bool,
    param_layout: Vec<(String, Vec<usize>, usize, usize)>,
    projected: GpuBuffer,
    convolved: GpuBuffer,
    scan: GpuBuffer,
    gated: GpuBuffer,
    states: GpuBuffer,
    grad_gated: GpuBuffer,
    grad_projected: GpuBuffer,
    grad_convolved: GpuBuffer,
    grad_scan: GpuBuffer,
    grad_state: GpuBuffer,
    no_cache: GpuBuffer,
    masked_input: GpuBuffer,
    scaled_input: GpuBuffer,
    pub(super) attention_input: GpuBuffer,
    pub(super) output: GpuBuffer,
    pub(super) combined: GpuBuffer,
    pub(super) grad_input: GpuBuffer,
    grad_output: GpuBuffer,
    pub(super) grad_branch: GpuBuffer,
    pub(super) attn_in: f32,
    pub(super) attn_out: f32,
    pub(super) key: f32,
    pub(super) mlp_gate: f32,
    pub(super) mlp_down: f32,
    ssm_in: f32,
    ssm_out: f32,
}

pub(super) fn scale(cmd: &mut vulkan::ComputeBatch, k: &TransformerKernels,
    input: &GpuBuffer, output: &GpuBuffer, len: usize, value: f32) -> Result<()> {
    k.scale_copy.record_dispatch(cmd, &[input, output],
        bytemuck::bytes_of(&ScalePush { len: len as u32, scale: value }),
        [div_ceil_u32(len,256),1,1])
}

impl Mixer {
    fn new(device: &VulkanDevice, rows: usize, seq: usize, d: usize,
        cfg: &serde_json::Value, store: &mut Gpt2TensorStore, layer: usize) -> Result<Self> {
        let integer = |key: &str, default: usize| -> Result<usize> {
            let n=match cfg.get(key).filter(|v|!v.is_null()) {
                Some(v)=>v.as_u64().with_context(||format!("Falcon H1 {key} must be an integer"))? as usize,
                None=>default,
            };
            if n==0 || n>u32::MAX as usize { bail!("Falcon H1 {key} must be a positive u32"); }
            Ok(n)
        };
        let number = |key: &str, default: f32| -> Result<f32> {
            let v=cfg.get(key).and_then(|v|v.as_f64()).unwrap_or(default as f64) as f32;
            if !v.is_finite(){bail!("Falcon H1 {key} must be finite");} Ok(v)
        };
        let flag=|key: &str, default: bool| cfg.get(key).and_then(|v|v.as_bool()).unwrap_or(default);
        let inner=integer("mamba_d_ssm", integer("mamba_expand",2)?.checked_mul(d).context("Mamba width overflow")?)?;
        let heads=integer("mamba_n_heads",128)?;
        let groups=integer("mamba_n_groups",1)?;
        let n=integer("mamba_d_state",256)?;
        let conv_kernel=integer("mamba_d_conv",4)?;
        let chunk=integer("mamba_chunk_size",256)?;
        if inner%heads!=0 || heads%groups!=0 || inner%groups!=0 {bail!("Falcon H1 invalid Mamba head/group geometry");}
        if let Some(hd)=cfg.get("mamba_d_head").and_then(|v|v.as_u64()) {
            if hd as usize!=inner/heads {bail!("Falcon H1 mamba_d_head * mamba_n_heads != mamba_d_ssm");}
        }
        let cd=inner.checked_add(2*groups*n).context("Mamba conv width overflow")?;
        let width=inner+cd+heads;
        let multipliers=|key: &str,count: usize| -> Result<Vec<f32>> {
            match cfg.get(key).filter(|v|!v.is_null()) {
                None=>Ok(vec![1.0;count]),
                Some(v)=>{
                    let a=v.as_array().filter(|a|a.len()==count).with_context(||format!("Falcon H1 {key} must contain {count} multipliers"))?;
                    a.iter().map(|v|v.as_f64().map(|v|v as f32).filter(|v|v.is_finite()).context("non-finite multiplier")).collect()
                }
            }
        };
        let mup=multipliers("ssm_multipliers",5)?;
        let mlp=multipliers("mlp_multipliers",2)?;
        let dt_limits=cfg.get("time_step_limit").and_then(|v|v.as_array());
        let dt_min=dt_limits.and_then(|a|a.first()).and_then(|v|v.as_f64()).unwrap_or(0.0) as f32;
        let dt_max=dt_limits.and_then(|a|a.get(1)).and_then(|v|v.as_f64()).unwrap_or(f64::INFINITY) as f32;
        if dt_min<0.0 || !dt_min.is_finite() || dt_max<dt_min {bail!("Falcon H1 invalid time_step_limit");}
        let prefix=format!("model.layers.{layer}.mamba");
        let linear=|store: &mut Gpt2TensorStore,name: &str, input: usize, output: usize,bias: bool| -> Result<VulkanLinear> {
            let w=store.take_expected(&format!("{prefix}.{name}.weight"),&[output,input])?;
            if bias {
                VulkanLinear::new(device,input,output,&w,&store.take_expected(&format!("{prefix}.{name}.bias"),&[output])?)
            }else{VulkanLinear::new_no_bias(device,input,output,&w)}
        };
        let in_proj=linear(store,"in_proj",d,width,flag("mamba_proj_bias",false))?;
        let out_proj=linear(store,"out_proj",inner,d,flag("projectors_bias",false))?;
        let mut weights=Vec::new();
        let mut param_layout=Vec::new();
        for (name,shape,exists,default) in [
            ("conv1d.weight",vec![cd,1,conv_kernel],true,0.0),
            ("conv1d.bias",vec![cd],flag("mamba_conv_bias",true),0.0),
            ("A_log",vec![heads],true,0.0), ("D",vec![heads],true,1.0),
            ("dt_bias",vec![heads],true,0.0),
            ("norm.weight",vec![inner],flag("mamba_rms_norm",false),1.0),
        ] {
            let len=shape.iter().product(); let offset=weights.len();
            if exists {
                weights.extend(store.take_expected(&format!("{prefix}.{name}"),&shape)?);
                param_layout.push((name.to_owned(),shape,offset,len));
            }else{weights.extend(vec![default;len]);}
        }
        let zeros=|len|GpuBuffer::zeros_f32(device,len);
        Ok(Self {
            push:Push{mode:0,batch:(rows/seq) as u32,seq:seq as u32,inner:inner as u32,
                heads:heads as u32,groups:groups as u32,state_dim:n as u32,kernel:conv_kernel as u32,
                chunk:chunk as u32,norm:flag("mamba_rms_norm",false) as u32,norm_before:flag("mamba_norm_before_gate",true) as u32,cached:0,
                eps:number("rms_norm_eps",1e-5)?,dt_min,dt_max,mz:mup[0],mx:mup[1],mb:mup[2],mc:mup[3],mdt:mup[4],
                conv_bias:flag("mamba_conv_bias",true) as u32,_pad:[0;3]},
            kernel:vulkan::ComputeKernel::new(device,include_bytes!("../../shaders/falcon_h1.spv"),14,std::mem::size_of::<Push>() as u32)?,
            in_proj,out_proj,params:TrainableParameter::new(device,&weights,true)?,train_params:true,param_layout,
            projected:zeros(rows*width)?,convolved:zeros(rows*cd)?,scan:zeros(rows*inner)?,gated:zeros(rows*inner)?,
            states:zeros((rows+rows/seq)*inner*n)?,grad_gated:zeros(rows*inner)?,grad_projected:zeros(rows*width)?,
            grad_convolved:zeros(rows*cd)?,grad_scan:zeros(rows*inner)?,no_cache:zeros(1)?,
            grad_state:zeros(rows/seq*inner*n)?,
            masked_input:zeros(rows*d)?,scaled_input:zeros(rows*d)?,attention_input:zeros(rows*d)?,
            output:zeros(rows*d)?,combined:zeros(rows*d)?,grad_input:zeros(rows*d)?,grad_output:zeros(rows*d)?,grad_branch:zeros(rows*d)?,
            attn_in:number("attention_in_multiplier",1.0)?,attn_out:number("attention_out_multiplier",1.0)?,key:number("key_multiplier",1.0)?,
            mlp_gate:mlp[0],mlp_down:mlp[1],ssm_in:number("ssm_in_multiplier",1.0)?,ssm_out:number("ssm_out_multiplier",1.0)?,
        })
    }

    fn dispatch(&self,cmd: &mut vulkan::ComputeBatch,mask: &GpuBuffer, cache: Option<&GpuBuffer>,mode:u32,batch:usize,seq:usize,len:usize)->Result<()> {
        let mut p=self.push;p.mode=mode;p.batch=batch as u32;p.seq=seq as u32;p.cached=cache.is_some() as u32;
        self.kernel.record_dispatch(cmd,&[&self.projected,&self.params.values,&self.convolved,&self.scan,&self.gated,&self.states,
            &self.grad_gated,&self.grad_projected,&self.grad_convolved,&self.grad_scan,self.params.grad_buffer()?,cache.unwrap_or(&self.no_cache),mask,&self.grad_state],
            bytemuck::bytes_of(&p),[div_ceil_u32(len,64),1,1])
    }

    pub(super) fn forward(&self,cmd:&mut vulkan::ComputeBatch,k:&TransformerKernels,input:&GpuBuffer,mask:&GpuBuffer,batch:usize,seq:usize,cache:Option<&GpuBuffer>,training:bool,rng_step:u32,rng_seed:u32)->Result<()> {
        if cache.is_some() && std::env::var_os("HIERARCHOS_FALCON_H1_TRACE_CACHE").is_some() {
            eprintln!("Falcon H1 native recurrent dispatch: batch={batch}, seq={seq}");
        }
        let rows=batch*seq;let d=self.in_proj.input_dim;
        let source=if seq>1 {record_row_scale(cmd,k,input,mask,&self.masked_input,rows,d)?;&self.masked_input}else{input};
        scale(cmd,k,source,&self.scaled_input,rows*d,self.ssm_in)?;
        record_transformer_linear_forward(&self.in_proj,cmd,k,&self.scaled_input,&self.projected,rows,training,rng_step,rng_seed)?;
        self.dispatch(cmd,mask,cache,0,batch,seq,batch*self.conv_dim())?;
        self.dispatch(cmd,mask,cache,1,batch,seq,batch*self.push.inner as usize)?;
        self.dispatch(cmd,mask,cache,2,batch,seq,rows*self.push.inner as usize)?;
        record_transformer_linear_forward(&self.out_proj,cmd,k,&self.gated,&self.output,rows,training,rng_step,rng_seed)?;
        scale(cmd,k,&self.output,&self.output,rows*d,self.ssm_out)
    }
    fn conv_dim(&self)->usize {(self.push.inner+2*self.push.groups*self.push.state_dim) as usize}
    pub(super) fn cache_len(&self)->usize {self.conv_dim()*self.push.kernel as usize+(self.push.inner*self.push.state_dim) as usize}
    pub(super) fn prefill_cache(&self,cmd:&mut vulkan::ComputeBatch,mask:&GpuBuffer,cache:&GpuBuffer,seq:usize)->Result<()> {
        self.dispatch(cmd,mask,None,1,1,seq,self.push.inner as usize)?;
        self.dispatch(cmd,mask,Some(cache),8,1,seq,self.cache_len())
    }
    pub(super) fn backward(&self,cmd:&mut vulkan::ComputeBatch,k:&TransformerKernels,mask:&GpuBuffer,grad:&GpuBuffer,batch:usize,seq:usize)->Result<()> {
        let rows=batch*seq;let d=self.in_proj.input_dim;
        scale(cmd,k,grad,&self.grad_output,rows*d,self.ssm_out)?;
        self.out_proj.record_backward(cmd,k,&self.gated,&self.grad_output,&self.grad_gated,rows)?;
        self.dispatch(cmd,mask,None,3,batch,seq,batch*self.push.inner as usize)?;
        self.dispatch(cmd,mask,None,4,batch,seq,rows*self.push.inner as usize)?;
        self.dispatch(cmd,mask,None,5,batch,seq,self.push.groups as usize)?;
        self.dispatch(cmd,mask,None,6,batch,seq,self.conv_dim())?;
        self.dispatch(cmd,mask,None,7,batch,seq,self.push.inner as usize)?;
        self.in_proj.record_backward(cmd,k,&self.scaled_input,&self.grad_projected,&self.grad_input,rows)?;
        scale(cmd,k,&self.grad_input,&self.grad_input,rows*d,self.ssm_in)?;
        if seq>1 {record_row_scale(cmd,k,&self.grad_input,mask,&self.grad_input,rows,d)?;}
        Ok(())
    }
    pub(super) fn step(&self,cmd:&mut vulkan::ComputeBatch,k:&TransformerKernels,step:u32,hyper:AdamWHyperParams)->Result<()> {
        self.in_proj.record_step(cmd,k,step,hyper)?;
        self.out_proj.record_step(cmd,k,step,hyper)?;
        if self.train_params { self.params.record_step(cmd,k,step,hyper)?; }
        Ok(())
    }

    pub(super) fn freeze_base_parameters(&mut self) {
        self.in_proj.freeze_base();
        self.out_proj.freeze_base();
        self.train_params = false;
        // The fused backward ABI still writes scratch parameter adjoints.
        self.params.release_optimizer_state_keep_gradient();
    }
}

impl VulkanTransformerLayer {
    pub(super) fn falcon_peft_linears(&self, index: usize) -> Result<Vec<(String, &VulkanLinear)>> {
        let p = format!("model.layers.{index}");
        let m = self.falcon_h1.as_ref().context("missing Falcon H1 mixer")?;
        Ok(vec![
            (format!("{p}.self_attn.q_proj"), self.q_proj.as_ref().context("missing Falcon q")?),
            (format!("{p}.self_attn.k_proj"), self.k_proj.as_ref().context("missing Falcon k")?),
            (format!("{p}.self_attn.v_proj"), self.v_proj.as_ref().context("missing Falcon v")?),
            (format!("{p}.self_attn.o_proj"), &self.c_proj),
            (format!("{p}.feed_forward.gate_proj"), self.c_gate.as_ref().context("missing Falcon gate")?),
            (format!("{p}.feed_forward.up_proj"), &self.c_fc),
            (format!("{p}.feed_forward.down_proj"), &self.c_mlp_proj),
            (format!("{p}.mamba.in_proj"), &m.in_proj),
            (format!("{p}.mamba.out_proj"), &m.out_proj),
        ])
    }

    pub(super) fn falcon_peft_linears_mut(&mut self, index: usize) -> Result<Vec<(String, &mut VulkanLinear)>> {
        let p = format!("model.layers.{index}");
        let m = self.falcon_h1.as_mut().context("missing Falcon H1 mixer")?;
        Ok(vec![
            (format!("{p}.self_attn.q_proj"), self.q_proj.as_mut().context("missing Falcon q")?),
            (format!("{p}.self_attn.k_proj"), self.k_proj.as_mut().context("missing Falcon k")?),
            (format!("{p}.self_attn.v_proj"), self.v_proj.as_mut().context("missing Falcon v")?),
            (format!("{p}.self_attn.o_proj"), &mut self.c_proj),
            (format!("{p}.feed_forward.gate_proj"), self.c_gate.as_mut().context("missing Falcon gate")?),
            (format!("{p}.feed_forward.up_proj"), &mut self.c_fc),
            (format!("{p}.feed_forward.down_proj"), &mut self.c_mlp_proj),
            (format!("{p}.mamba.in_proj"), &mut m.in_proj),
            (format!("{p}.mamba.out_proj"), &mut m.out_proj),
        ])
    }
}

pub(super) fn attach(model:&mut VulkanTransformer,path:&Path)->Result<()> {
    let cfg=transformer_text_config_from_package(path)?;
    let mut store=load_llama_layout_tensor_store(path)?;
    model.kernels.adamw = vulkan::ComputeKernel::new(&model.device,
        include_bytes!("../../shaders/falcon_h1_adamw.spv"), 4, 28)?;
    model.final_norm.unfused_products=true;
    // The local HF/PyTorch FP32 outer RMSNorm uses an eight-lane mean
    // reduction. Scalar association perturbs near-cancelling full-training
    // gradients enough for eps=1e-8 AdamW to amplify them above the gate.
    model.final_norm.extra_rsqrt_refinement=3;
    for (index,layer) in model.layers.iter_mut().enumerate(){
        layer.ln1.unfused_products=true;
        layer.ln2.unfused_products=true;
        layer.ln1.extra_rsqrt_refinement=3;
        layer.ln2.extra_rsqrt_refinement=3;
        layer.falcon_h1=Some(Mixer::new(&model.device,model.batch_size*model.seq_len,model.seq_len,model.config.hidden_size,&cfg,&mut store,index)?);
    }
    Ok(())
}

pub(super) fn export(model:&VulkanTransformer)->Result<BTreeMap<String,(Vec<usize>,Vec<f32>)>> {
    let mut values=model.llama_parameter_tensors()?;
    let final_norm=values.remove("model.norm.weight").context("Falcon H1 final norm missing")?;
    values.insert("model.final_layernorm.weight".into(),final_norm);
    for (index,layer) in model.layers.iter().enumerate(){
        let prefix=format!("model.layers.{index}");
        let norm=values.remove(&format!("{prefix}.post_attention_layernorm.weight")).context("Falcon H1 pre-FF norm missing")?;
        values.insert(format!("{prefix}.pre_ff_layernorm.weight"),norm);
        for name in ["gate_proj","up_proj","down_proj"] {
            for suffix in ["weight","bias"] {
                if let Some(tensor)=values.remove(&format!("{prefix}.mlp.{name}.{suffix}")) {
                    values.insert(format!("{prefix}.feed_forward.{name}.{suffix}"),tensor);
                }
            }
        }
        let m=layer.falcon_h1.as_ref().context("Falcon H1 Mamba branch missing")?;
        for (name,linear) in [("in_proj",&m.in_proj),("out_proj",&m.out_proj)]{
            values.insert(format!("{prefix}.mamba.{name}.weight"),(vec![linear.output_dim,linear.input_dim],linear.weight.read()?));
            if linear.has_bias{values.insert(format!("{prefix}.mamba.{name}.bias"),(vec![linear.output_dim],linear.bias.read()?));}
        }
        let params=m.params.read()?;
        for (name,shape,start,len) in &m.param_layout {
            values.insert(format!("{prefix}.mamba.{name}"),(shape.clone(),params[*start..start+len].to_vec()));
        }
    }
    Ok(values)
}

impl VulkanTransformer {
    /// Read the most recent Falcon H1 tape for strict oracle diagnostics.
    #[doc(hidden)]
    pub fn debug_falcon_h1_last_forward_states(&self)->Result<BTreeMap<String,Vec<f32>>> {
        if self.config.architecture!=VulkanTransformerArchitecture::FalconH1 {bail!("Falcon H1 trace requires falcon_h1");}
        let mut out=BTreeMap::new();let r=self.rows;let d=self.config.hidden_size;
        out.insert("embed".into(),self.input_hidden.read_f32(r*d)?);
        out.insert("final_norm".into(),self.final_norm_output.read_f32(r*d)?);
        out.insert("logits".into(),self.logits.read_f32(r*self.config.vocab_size)?);
        for (i,l) in self.layers.iter().enumerate(){
            let m=l.falcon_h1.as_ref().context("missing Mamba branch")?;
            out.insert(format!("layers.{i}.ln1_flags"),vec![f32::from(l.ln1.unfused_products),l.ln1.extra_rsqrt_refinement as f32]);
            out.insert(format!("layers.{i}.ln1_mean"),l.tape.ln1_mean.read_f32(r)?);
            out.insert(format!("layers.{i}.ln1_rstd"),l.tape.ln1_rstd.read_f32(r)?);
            for (name,buffer,len) in [
                ("ln1",&l.tape.ln1,r*d),("mamba.input",&m.scaled_input,r*d),
                ("mamba.in_proj",&m.projected,r*m.in_proj.output_dim),
                ("mamba.conv",&m.convolved,r*m.conv_dim()),("mamba.scan",&m.scan,r*m.push.inner as usize),
                ("mamba.gated",&m.gated,r*m.push.inner as usize),("mamba.output",&m.output,r*d),
                ("q",&l.tape.q,r*l.query_hidden_size),("k",&l.tape.k,r*l.key_hidden_size),
                ("v",&l.tape.v,r*l.value_hidden_size),("q_rope",&l.tape.q_rotary,r*l.query_hidden_size),
                ("k_rope",&l.tape.k_rotary,r*l.key_hidden_size),("attention",&l.tape.attention,r*l.attention_hidden_size),
                ("attention.output",&l.tape.attention_projection,r*d),("residual",&l.tape.residual1,r*d),
                ("ln2",&l.tape.ln2,r*d),("mlp.gate",&l.tape.mlp_gate_pre,r*l.c_fc.output_dim),
                ("mlp.output",&l.tape.mlp_output,r*d),("output",&l.tape.output,r*d),
            ] {out.insert(format!("layers.{i}.{name}"),buffer.read_f32(len)?);}
        }
        Ok(out)
    }

    /// Read Falcon H1 gradients retained after the most recent native training step.
    #[doc(hidden)]
    pub fn debug_falcon_h1_last_backward_states(&self)->Result<BTreeMap<String,Vec<f32>>> {
        if self.config.architecture!=VulkanTransformerArchitecture::FalconH1 {bail!("Falcon H1 backward trace requires falcon_h1");}
        let mut out=BTreeMap::new();let r=self.rows;let d=self.config.hidden_size;
        out.insert("grad_logits".into(),self.grad_logits.read_f32(r*self.config.vocab_size)?);
        out.insert("grad_final_norm_output".into(),self.grad_final_norm_output.read_f32(r*d)?);
        out.insert("grad_final_norm_input".into(),self.grad_final_norm_input.read_f32(r*d)?);
        for (i,l) in self.layers.iter().enumerate(){
            let m=l.falcon_h1.as_ref().context("missing Mamba branch")?;
            let prefix=format!("layers.{i}");
            for (name,buffer) in [
                ("grad_ln2",&l.tape.grad_ln2),("grad_ln1",&l.tape.grad_ln1),
                ("grad_residual1_mlp",&l.tape.grad_residual1_mlp),
                ("grad_input",&l.tape.grad_input),
            ] {out.insert(format!("{prefix}.{name}"),buffer.read_f32(r*d)?);}
            out.insert(format!("{prefix}.mamba.gated"),m.gated.read_f32(r*m.push.inner as usize)?);
            out.insert(format!("{prefix}.mamba.out_proj_weight_grad"),m.out_proj.weight.grad_buffer()?.read_f32(m.out_proj.weight.len)?);
            out.insert(format!("{prefix}.mamba.in_proj_weight_grad"),m.in_proj.weight.grad_buffer()?.read_f32(m.in_proj.weight.len)?);
            out.insert(format!("{prefix}.mamba.grad_output"),m.grad_output.read_f32(r*d)?);
            out.insert(format!("{prefix}.mamba.grad_input"),m.grad_input.read_f32(r*d)?);
        }
        Ok(out)
    }
}

//! Small standalone shader/I/O probe; includes the production Vulkan runtime.
#![allow(dead_code)]
#[path = "../src/vulkan.rs"] mod vulkan;
#[cfg(not(baseline_io))]
#[path = "../src/weight_offload.rs"] mod weight_offload;
#[cfg(baseline_io)]
#[path = "../../target/gen9-phase4/baseline-weight_offload.rs"] mod weight_offload;
pub use vulkan::VulkanDevice;
use vulkan::{ComputeBatch, ComputeKernel, GpuBuffer};
use anyhow::{bail, Result};
use std::{path::Path, time::Instant};

fn values(n:usize,salt:usize)->Vec<f32> {
    (0..n).map(|i|((i*73+salt*31)%251) as f32 / 127.0 - 0.98).collect()
}
fn bytes(words:&[u32])->&[u8] { bytemuck::cast_slice(words) }
fn kernel(device:&VulkanDevice,path:&Path,bindings:usize,push:u32)->Result<ComputeKernel> {
    ComputeKernel::new(device,&std::fs::read(path)?,bindings,push)
}
fn execute(device:&VulkanDevice,kernel:&ComputeKernel,buffers:&[&GpuBuffer],p:&[u32],grid:[u32;3],repeats:usize)->Result<f64> {
    let start=Instant::now();
    let mut batch=ComputeBatch::new(device)?;
    for _ in 0..repeats { kernel.record_dispatch(&mut batch,buffers,bytes(p),grid)?; }
    batch.submit()?;
    Ok(start.elapsed().as_secs_f64()*1000.0/repeats as f64)
}
fn qsa(root:&Path)->Result<()> {
    let device=VulkanDevice::new()?;
    let old=kernel(&device,&root.join("baseline-shaders/transformer_qwen4_qsa_mask.spv"),7,96)?;
    let new=kernel(&device,&root.join("candidate-shaders/transformer_qwen4_qsa_mask_gen9.spv"),7,96)?;
    let mut records=Vec::new();
    for (batch,query,key,ratio,topk,offset) in [
        (1,6,6,2,2,0),(2,9,17,2,2,0),(1,1,33,4,2,32),
        (1,1,65,8,4,64),(2,7,19,3,2,4),(1,3,7,4,0,4),
        (1,3,7,1,7,0),(1,1,257,1,0,256),(1,1,257,2,128,256),
        (1,1,257,2,6,256),
    ] {
        for (heads,dim) in [(1,4),(2,8),(3,16),(17,4)] {
            for mask_mode in 0..4 {
                if key > 100 && topk == 6 && mask_mode != 3 { continue; }
                let packed=heads*dim+dim;
                let mut q=values(batch*query*packed,17);
                let mut k=values(batch*key*dim,23);
                if mask_mode==3 {q.fill(0.0); k.fill(0.0);}
                let mask=(0..batch*key).map(|i|if mask_mode==1 && i%5==0 || mask_mode==2 && i%7>3 {0.0}else{1.0}).collect::<Vec<_>>();
                let q=GpuBuffer::from_f32(&device,&q)?;
                let k=GpuBuffer::from_f32(&device,&k)?;
                let mask=GpuBuffer::from_f32(&device,&mask)?;
                let norm=GpuBuffer::from_f32(&device,&values(dim,47).iter().map(|x|x*0.02).collect::<Vec<_>>())?;
                let factors=GpuBuffer::from_f32(&device,&vec![1.0;dim])?;
                let output=GpuBuffer::zeros_f32(&device,batch*query*key)?;
                let p=[batch as u32,query as u32,key as u32,heads as u32,dim as u32,packed as u32,
                    ratio as u32,topk as u32,offset as u32,1,dim as u32,0,10000.0f32.to_bits(),
                    0,1.0f32.to_bits(),128,1.0f32.to_bits(),4.0f32.to_bits(),1.0f32.to_bits(),
                    32.0f32.to_bits(),1.0f32.to_bits(),1,0,1e-6f32.to_bits()];
                let buffers=[&q,&k,&mask,&norm,&norm,&factors,&output];
                // Very long all-selected controls have an analytical mask and
                // avoid exercising the old quadratic selector past its safe size.
                let reference=if key>100 {
                    let masks=mask.read_f32(batch*key)?;
                    let count=masks.iter().filter(|&&x|x>0.0).count();
                    if topk>=count/ratio {masks} else if topk==0 && ratio==1 {vec![0.0;batch*key]}
                    else if mask_mode == 3 {
                        (0..key).map(|i| if i < topk*ratio || i >= count/ratio*ratio {1.0}else{0.0}).collect()
                    } else {continue}
                } else {
                    execute(&device,&old,&buffers,&p,[1,(query as u32+7)/8,batch as u32],1)?;
                    output.read_f32(batch*query*key)?
                };
                let mut baseline_ms=None;
                let repeats=if key<40 {11}else{1};
                if key<40 {baseline_ms=Some(execute(&device,&old,&buffers,&p,[1,(query as u32+7)/8,batch as u32],repeats)?);}
                let candidate_ms=execute(&device,&new,&buffers,&p,[1,(query as u32+7)/8,batch as u32],repeats)?;
                let actual=output.read_f32(batch*query*key)?;
                if actual.iter().map(|x|x.to_bits()).ne(reference.iter().map(|x|x.to_bits())) {
                    bail!("QSA mismatch batch={batch} query={query} key={key} heads={heads} dim={dim} mask={mask_mode}");
                }
                records.push(serde_json::json!({"batch":batch,"query":query,"key":key,"heads":heads,
                    "dim":dim,"mask":mask_mode,"ratio":ratio,"topk":topk,"baseline_ms":baseline_ms,
                    "candidate_ms":candidate_ms,"words":actual.len(),"bit_exact":true}));
            }
        }
    }
    println!("{}",serde_json::json!({"device":device.name(),"cases":records.len(),"records":records}));
    Ok(())
}
fn hc(root:&Path)->Result<()> {
    let device=VulkanDevice::new()?;
    let old=kernel(&device,&root.join("baseline-shaders/transformer_deepseek_v41_hc_split.spv"),7,28)?;
    let new=kernel(&device,&root.join("candidate-shaders/transformer_deepseek_v41_hc_split_gen9.spv"),7,28)?;
    let mut records=Vec::new();
    for rows in [1,3,8] {for hc in [1,2,4,8,16] {for dim in [3,16,65] {for iters in [1,4,20] {
        let mix_hc=hc*2+hc*hc;
        let stream=GpuBuffer::from_f32(&device,&values(rows*hc*dim,3))?;
        let mixes=GpuBuffer::from_f32(&device,&values(rows*mix_hc,7))?;
        let scale=GpuBuffer::from_f32(&device,&[0.5,0.7,0.11])?;
        let base=GpuBuffer::from_f32(&device,&values(mix_hc,13))?;
        let pre=GpuBuffer::zeros_f32(&device,rows*hc)?;
        let post=GpuBuffer::zeros_f32(&device,rows*hc)?;
        let comb=GpuBuffer::zeros_f32(&device,rows*hc*hc)?;
        let p=[rows as u32,hc as u32,dim as u32,mix_hc as u32,iters,1e-6f32.to_bits(),1e-6f32.to_bits()];
        let buffers=[&stream,&mixes,&scale,&base,&pre,&post,&comb];
        let baseline_ms=execute(&device,&old,&buffers,&p,[rows as u32,1,1],7)?;
        let expected=[pre.read_f32(rows*hc)?,post.read_f32(rows*hc)?,comb.read_f32(rows*hc*hc)?].concat();
        let candidate_ms=execute(&device,&new,&buffers,&p,[rows as u32,1,1],7)?;
        let actual=[pre.read_f32(rows*hc)?,post.read_f32(rows*hc)?,comb.read_f32(rows*hc*hc)?].concat();
        if actual.iter().map(|x|x.to_bits()).ne(expected.iter().map(|x|x.to_bits())) {bail!("HC mismatch rows={rows} hc={hc} dim={dim} iters={iters}");}
        records.push(serde_json::json!({"rows":rows,"hc":hc,"dim":dim,"iters":iters,
            "baseline_ms":baseline_ms,"candidate_ms":candidate_ms,"words":actual.len(),"bit_exact":true}));
    }}}}
    println!("{}",serde_json::json!({"device":device.name(),"cases":records.len(),"records":records}));
    Ok(())
}
fn silu(root:&Path)->Result<()> {
    let device=VulkanDevice::new()?;
    let old=kernel(&device,&root.join("baseline-shaders/silu_forward.spv"),2,4)?;
    let mul=kernel(&device,&root.join("../../hierarchos-vulkan/shaders/transformer_elementwise_mul_forward.spv"),3,4)?;
    let new=kernel(&device,&root.join("candidate-shaders/silu_forward_gen9.spv"),4,4)?;
    let mut records=Vec::new();
    for n in [1usize,17,63,64,65,128,255,256,257,768,3072,16384,65280] {
        let inputs=if n==65280 {(0..65536u32).map(|x|f32::from_bits(x<<16)).filter(|x|x.is_finite()).collect::<Vec<_>>()}
            else {values(n,3).iter().map(|x|x*10.0).collect::<Vec<_>>()};
        let input=GpuBuffer::from_f32(&device,&inputs)?;
        let up=GpuBuffer::from_f32(&device,&values(n,9))?;
        let activation=GpuBuffer::zeros_f32(&device,n)?;
        let product=GpuBuffer::zeros_f32(&device,n)?;
        let p=[n as u32];
        let start=Instant::now();
        let mut batch=ComputeBatch::new(&device)?;
        for _ in 0..31 {
            old.record_dispatch(&mut batch,&[&input,&activation],bytes(&p),[(n as u32+255)/256,1,1])?;
            mul.record_dispatch(&mut batch,&[&activation,&up,&product],bytes(&p),[(n as u32+255)/256,1,1])?;
        }
        batch.submit()?;
        let baseline_ms=start.elapsed().as_secs_f64()*1000.0/31.0;
        let expected=[activation.read_f32(n)?,product.read_f32(n)?].concat();
        let candidate_ms=execute(&device,&new,&[&input,&activation,&up,&product],&p,[(n as u32+63)/64,1,1],31)?;
        let actual=[activation.read_f32(n)?,product.read_f32(n)?].concat();
        if actual.iter().map(|x|x.to_bits()).ne(expected.iter().map(|x|x.to_bits())) {bail!("SiLU product mismatch n={n}");}
        records.push(serde_json::json!({"elements":n,"words":actual.len(),"baseline_ms":baseline_ms,"candidate_ms":candidate_ms,"bit_exact":true}));
    }
    println!("{}",serde_json::json!({"device":device.name(),"cases":records.len(),"records":records}));
    Ok(())
}
fn io(root:&Path)->Result<()> {
    let path=root.join("../gen9-phase3/cpu-fixtures-reproduced/dbrx/unmasked/model");
    let mut package=weight_offload::ModelPackage::open_directory(path)?.with_cache_bytes(0);
    let names=package.tensor_names()?;
    let mut records=Vec::new();
    for name in names {
        let shape=package.shape(&name)?;
        let count=shape.iter().product::<usize>();
        if shape.len()!=2 || count<128 {continue;}
        let warm=package.read_f32_elements(&name,0,count.min(512))?;
        let start=Instant::now();
        for _ in 0..300 {
            let actual=package.read_f32_elements(&name,0,count.min(512))?;
            assert_eq!(actual,warm);
        }
        records.push(serde_json::json!({"name":name,"ms":start.elapsed().as_secs_f64()*1000.0,"elements":warm.len()}));
    }
    println!("{}",serde_json::json!({"records":records,"ledger":package.ledger()}));
    Ok(())
}
fn main()->Result<()> {
    let mut args=std::env::args().skip(1);
    let command=args.next().unwrap_or_default();
    let root=args.next().unwrap_or_else(||"target/gen9-phase4".into());
    match command.as_str() {"qsa"=>qsa(Path::new(&root)),"hc"=>hc(Path::new(&root)),"silu"=>silu(Path::new(&root)),"io"=>io(Path::new(&root)),_=>bail!("expected qsa|hc|silu|io [PHASE_DIR]")}
}

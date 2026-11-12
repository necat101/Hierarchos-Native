use super::*;
use hierarchos_vulkan::{VulkanImageText,VulkanLlava,VulkanGemma3Vision,VulkanGemma4Vision};
use hierarchos_vulkan::deepseek_v41::{DeepseekV41AdamWConfig,DeepseekV41Checkpoint,DeepseekV41LoraAdapter,DeepseekV41ModelTrainingConfig,DeepseekV41MultimodalTrainer,DeepseekV41TrainingBatch};

pub(super) enum Runtime {
    Shared(Box<VulkanImageText>),Llava(Box<VulkanLlava>),Gemma3(Box<VulkanGemma3Vision>),
    Gemma4(Box<VulkanGemma4Vision>),V41(Box<DeepseekV41MultimodalTrainer>),
}
impl Runtime {
    pub fn new(p:&Package,picture:&Picture,o:&Options)->anyhow::Result<Self> {
        let wmma=hierarchos_vulkan::wmma::WmmaMode::from_env()?;
        anyhow::ensure!(wmma!=hierarchos_vulkan::wmma::WmmaMode::FullFp16,"full-FP16 is unavailable for vision graphs; use --wmma-mode off or mixed");
        anyhow::ensure!(p.kind!=Kind::V41||!wmma.requested(),"native DeepSeek V4.1 vision uses FP32 kernels and has no WMMA training integration; use --wmma-mode off");
        anyhow::ensure!(p.kind==Kind::V41||(!o.stream&&o.state.is_none()&&o.resume.is_none()&&o.adapter.is_none()),"native state, resume, adapter and frozen-parameter streaming options currently apply only to native DeepSeek V4.1; select an exported image package for other families");
        if matches!(p.kind,Kind::Gemma4|Kind::V41){anyhow::ensure!(o.batch==1,"this native image backend currently requires batch size 1 (one complete image per trainer step)");}
        if let Some(state)=o.resume.as_ref().or(o.state.as_ref()) {
            let metadata=package::read_json(state.join("multimodal_state.json"))?;
            for field in ["stream_experts","stream_projections","stream_engram_tables","stream_vocabulary"] {
                anyhow::ensure!(metadata["options"][field].as_bool()==Some(o.stream),"--stream-parameters must match the recorded native state residency");
            }
            if o.resume.is_some() {
                let expected=serde_json::to_value(DeepseekV41AdamWConfig{learning_rate:o.hyper.lr,beta1:o.hyper.beta1,beta2:o.hyper.beta2,eps:o.hyper.eps,weight_decay:o.hyper.weight_decay})?;
                anyhow::ensure!(metadata["hyper"]==expected,"resume optimizer controls must match the saved checkpoint (lr, beta1, beta2, eps, weight_decay)");
            }
        }
        let device=if let Some(index)=o.device{VulkanDevice::new_with_index(index)?}else{VulkanDevice::new()?};
        eprintln!("Vision backend: {:?}; device: {}; precision: {}",p.kind,device.name(),wmma.description());
        let d=device;
        Ok(match p.kind {
            Kind::Shared=>Self::Shared(Box::new(VulkanImageText::from_hf_package(d,&p.root,o.batch,o.seq,o.batch)?)),
            Kind::Llava=>Self::Llava(Box::new(VulkanLlava::from_hf_package(d,&p.root,o.batch,o.seq,o.batch)?)),
            Kind::Gemma3=>Self::Gemma3(Box::new(VulkanGemma3Vision::from_hf_package(d,&p.root,o.batch,o.seq,o.batch)?)),
            Kind::Gemma4=>Self::Gemma4(Box::new(VulkanGemma4Vision::from_hf_package(d,&p.root,o.batch,o.seq,1,picture.width)?)),
            Kind::V41=>{
                let t=if let Some(path)=&o.resume {DeepseekV41MultimodalTrainer::resume(d,&p.root,path)?}
                else if let Some(path)=&o.state {DeepseekV41MultimodalTrainer::from_native_export(d,&p.root,path)?}
                else {
                    let mut checkpoint=DeepseekV41Checkpoint::open(&p.root)?;
                    let adapter=if let Some(path)=&o.adapter {DeepseekV41LoraAdapter::load(path)?}
                        else {DeepseekV41LoraAdapter::initialize(&mut checkpoint,&[],o.rank,o.alpha.unwrap_or((o.rank*2) as f64),false,o.seed)?};
                    let patch=positive(&p.config["vision_config"],"patch_size")?;
                    let hyper=DeepseekV41AdamWConfig{learning_rate:o.hyper.lr,beta1:o.hyper.beta1,beta2:o.hyper.beta2,eps:o.hyper.eps,weight_decay:o.hyper.weight_decay};
                    let config=DeepseekV41ModelTrainingConfig{batch_size:1,stream_experts:o.stream,stream_projections:o.stream,stream_engram_tables:o.stream,stream_vocabulary:o.stream,..Default::default()};
                    DeepseekV41MultimodalTrainer::new(d,&p.root,&adapter,hyper,config,picture.height/patch,picture.width/patch)?
                };Self::V41(Box::new(t))
            },
        })
    }
    fn pixels(&self,p:&Package,picture:&Picture,rgb:&[u8])->anyhow::Result<Vec<f32>> {
        match self {
            Self::Shared(g)=>g.preprocess_rgb8(rgb),Self::Llava(g)=>g.preprocess_rgb8(rgb),
            Self::Gemma3(g)=>g.preprocess_rgb8(rgb),Self::Gemma4(g)=>g.preprocess_rgb8(rgb),
            Self::V41(_)=>Ok(picture.patches(positive(&p.config["vision_config"],"patch_size")?)),
        }
    }
    fn v41_batch(p:&Package,picture:&Picture,b:&Batch,seq:usize)->anyhow::Result<(DeepseekV41TrainingBatch,Vec<i32>,usize)> {
        anyhow::ensure!(b.weights.iter().all(|&w|w==0.||w==1.),"V4.1 requires binary loss weights");
        let (types,start)=image_layout(p,picture,&b.ids,&b.mask)?;
        let batch=DeepseekV41TrainingBatch{token_ids:b.ids.clone(),targets:b.targets.iter().zip(&b.weights).map(|(&id,&w)|if w>0.{id}else{u32::MAX}).collect(),seq_len:seq,engram_mask:None};
        Ok((batch,types,start))
    }
    pub fn train(&mut self,p:&Package,samples:&[Sample],o:&Options)->anyhow::Result<f32> {
        let b=Batch::from_samples(samples,o.batch);let picture=&samples[0].picture;
        let pixels=self.pixels(p,picture,&b.rgb)?;
        Ok(match self {
            Self::Shared(g)=> if p.seq2seq {g.train_seq2seq_with_optimizer(&b.ids,&b.mask,&b.decoder,&b.decoder_mask,&pixels,&b.targets,&b.weights,o.hyper)?.text.loss}
                else {g.train_step_with_optimizer(&b.ids,&b.mask,&pixels,&b.targets,&b.weights,o.hyper)?.text.loss},
            Self::Llava(g)=>g.train_step_with_optimizer(&b.ids,&b.mask,&pixels,&b.targets,&b.weights,o.hyper)?.text.loss,
            Self::Gemma3(g)=>g.train_step_with_optimizer(&b.ids,&b.mask,&pixels,&b.targets,&b.weights,o.hyper)?.text.loss,
            Self::Gemma4(g)=>g.train_step_with_optimizer(&b.ids,&b.mask,&pixels,&b.targets,&b.weights,o.hyper)?.text.loss,
            Self::V41(g)=>{let (batch,types,start)=Self::v41_batch(p,picture,&b,o.seq)?;g.train_step(&batch,&pixels,&types,start)?.loss},
        })
    }
    pub fn forward(&mut self,p:&Package,picture:&Picture,b:&Batch,seq:usize)->anyhow::Result<Vec<f32>> {
        let pixels=self.pixels(p,picture,&b.rgb)?;
        Ok(match self {
            Self::Shared(g)=>if p.seq2seq {g.forward_seq2seq(&b.ids,&b.mask,&b.decoder,&b.decoder_mask,&pixels)?.logits}else{g.forward(&b.ids,&b.mask,&pixels)?.logits},
            Self::Llava(g)=>g.forward(&b.ids,&b.mask,&pixels)?.logits,
            Self::Gemma3(g)=>g.forward(&b.ids,&b.mask,&pixels)?.logits,
            Self::Gemma4(g)=>g.forward(&b.ids,&b.mask,&pixels)?.logits,
            Self::V41(g)=>{let (batch,types,start)=Self::v41_batch(p,picture,b,seq)?;g.forward_logits(&batch,&pixels,&types,start)?.read_all()?},
        })
    }
    pub fn export(&self,p:&Package,path:&Path)->anyhow::Result<()> {
        match self {
            Self::Shared(g)=>g.export_hf_package(&p.root,path),Self::Llava(g)=>g.export_hf_package(&p.root,path),
            Self::Gemma3(g)=>g.export_hf_package(&p.root,path),Self::Gemma4(g)=>g.export_hf_package(&p.root,path),Self::V41(g)=>g.export_native(path),
        }
    }
    pub fn checkpoint(&self,path:&Path)->anyhow::Result<()> {
        match self {Self::V41(g)=>g.save_checkpoint(path),_=>anyhow::bail!("optimizer resume checkpoints are available for native V4.1; other image families export trained model weights")}
    }
}

//! Dedicated V4.1 dispatch using the same native model trainer as qualification.
use super::*;
use hierarchos_vulkan::deepseek_v41::{DeepseekV41AdamWConfig,DeepseekV41Checkpoint,DeepseekV41LoraAdapter,DeepseekV41ModelTrainer,DeepseekV41ModelTrainingConfig,DeepseekV41TrainingBatch};

pub(super) struct Options {
    pub model_dir:PathBuf,pub dataset:PathBuf,pub output:PathBuf,pub checkpoint:Option<PathBuf>,pub resume:Option<PathBuf>,pub adapter:Option<PathBuf>,
    pub batch_size:usize,pub seq_len:usize,pub epochs:usize,pub device_index:Option<usize>,pub rank:usize,pub alpha:Option<usize>,pub targets:Vec<String>,pub rslora:bool,pub seed:u64,pub hyper:DeepseekV41AdamWConfig,pub stream_parameters:bool,
}
pub(super) fn run(o:Options)->Result<u8,String> {
    let result=(||->anyhow::Result<()> {
        for file in ["adapter_config.json","adapter_model.safetensors"] { if o.output.join(file).exists(){anyhow::bail!("adapter output already exists: {}",o.output.join(file).display());} }
        let mut package=DeepseekV41Checkpoint::open(&o.model_dir)?;
        let seq=o.seq_len.min(package.config.text.max_position_embeddings).min(65_535);
        if seq==0 {anyhow::bail!("V4.1 sequence length must be positive");}
        let hidden=package.config.text.hidden_size;
        let device=match o.device_index {Some(i)=>VulkanDevice::new_with_index(i)?,None=>VulkanDevice::new()?};
        let mut trainer=if let Some(path)=&o.resume {
            let t=DeepseekV41ModelTrainer::resume(device,&o.model_dir,path)?;
            if t.training_config().batch_size!=o.batch_size {anyhow::bail!("V4.1 resumed batch size differs from checkpoint");}
            let c=t.training_config();
            if [c.stream_experts,c.stream_projections,c.stream_engram_tables,c.stream_vocabulary].iter().any(|&v|v!=o.stream_parameters) {
                anyhow::bail!("V4.1 resume residency differs from checkpoint: --stream-parameters must match its recorded expert/projection/Engram/vocabulary streaming choices");
            }
            t
        } else {
            let adapter=if let Some(path)=&o.adapter {DeepseekV41LoraAdapter::load(path)?} else {DeepseekV41LoraAdapter::initialize(&mut package,&o.targets,o.rank,o.alpha.unwrap_or(o.rank.saturating_mul(2)) as f64,o.rslora,o.seed)?};
            DeepseekV41ModelTrainer::new(device,&o.model_dir,&adapter,o.hyper,DeepseekV41ModelTrainingConfig{batch_size:o.batch_size,stream_experts:o.stream_parameters,stream_projections:o.stream_parameters,stream_engram_tables:o.stream_parameters,stream_vocabulary:o.stream_parameters,..Default::default()})?
        };
        drop(package);
        let mut steps=0usize;let mut last=None;
        for epoch in 0..o.epochs {
            let mut pending=Vec::with_capacity(o.batch_size);
            for (line_no,line) in BufReader::new(File::open(&o.dataset)?).lines().enumerate() {
                let line=line?;if line.trim().is_empty(){continue;}
                let value:Value=serde_json::from_str(&line)?;
                for chunk in transformer_chunks_from_value(&value,seq,hidden).map_err(anyhow::Error::msg)? {
                    if chunk.encoder_context.is_some(){anyhow::bail!("V4.1 text training does not accept encoder context");}
                    let mut padded=false;
                    for &mask in &chunk.attention_mask {
                        if mask==0.0 {padded=true;} else if mask!=1.0 || padded {anyhow::bail!("V4.1 attention mask must be a visible prefix followed by padding (line {})",line_no+1);}
                    }
                    if chunk.loss_weights.iter().any(|&w|w!=0.0&&w!=1.0){anyhow::bail!("V4.1 mean cross entropy accepts binary loss masks (line {})",line_no+1);}
                    pending.push(chunk);
                    if pending.len()==o.batch_size {
                        let loss=submit(&mut trainer,&pending,o.batch_size,seq)?;
                        steps+=1;last=Some(loss);pending.clear();
                        eprintln!("{}",json!({"architecture":"deepseek_v41","epoch":epoch+1,"step":trainer.optimizer().step_count(),"loss":loss}));
                    }
                }
            }
            if !pending.is_empty(){last=Some(submit(&mut trainer,&pending,o.batch_size,seq)?);steps+=1;}
        }
        if last.is_none(){anyhow::bail!("V4.1 dataset has no supervised training chunks");}
        let nonce=SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let staged=o.output.join(format!(".v41-export-{}-{nonce}",std::process::id()));
        trainer.export_adapter(&staged)?;
        for file in ["adapter_config.json","adapter_model.safetensors"] {fs::rename(staged.join(file),o.output.join(file))?;}
        fs::remove_dir(staged)?;
        if let Some(path)=&o.checkpoint {trainer.save_checkpoint(path)?;}
        fs::write(o.output.join("training_report.json"),serde_json::to_vec_pretty(&json!({"backend":"native-vulkan","architecture":"deepseek_v41","steps":steps,"optimizer_step":trainer.optimizer().step_count(),"loss":last,"stream_experts":o.stream_parameters,"stream_parameters":o.stream_parameters,"base_sha256":trainer.base_fingerprint()}))?)?;
        Ok(())
    })();
    result.map(|_|0).map_err(|e|format!("DeepSeek V4.1 native training failed: {e:#}"))
}
fn submit(trainer:&mut DeepseekV41ModelTrainer,chunks:&[TransformerChunk],batch:usize,seq:usize)->anyhow::Result<f32> {
    let mut b=DeepseekV41TrainingBatch{token_ids:vec![0;batch*seq],targets:vec![u32::MAX;batch*seq],seq_len:seq,engram_mask:Some(vec![1;batch*seq])};
    for (row,c) in chunks.iter().enumerate() {
        let at=row*seq;b.token_ids[at..at+seq].copy_from_slice(&c.input_ids);
        for col in 0..seq {
            if c.loss_weights[col]>0.0 {b.targets[at+col]=c.targets[col];}
            b.engram_mask.as_mut().unwrap()[at+col]=u32::from(c.attention_mask[col]==0.0);
        }
    }
    Ok(trainer.train_step(&b)?.loss)
}

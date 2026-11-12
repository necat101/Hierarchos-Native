//! Native image/text commands used by the desktop GUI. No Python dispatcher.
use super::*;
use anyhow::{Context,Result as AResult};
use hierarchos_vulkan::CpuAdamWHyperParams;
mod package;
mod data;
mod runtime;
use package::{Package,Kind,Picture,positive};
use data::{Sample,Batch,prompt_ids,image_layout};
use runtime::Runtime;

struct Options {
    model:PathBuf,train:Option<PathBuf>,image:Option<PathBuf>,output:Option<PathBuf>,tokenizer:Option<PathBuf>,
    prompt:String,prompt_ids:Option<Vec<u32>>,batch:usize,seq:usize,epochs:usize,device:Option<usize>,
    hyper:CpuAdamWHyperParams,rank:usize,alpha:Option<f64>,seed:u64,stream:bool,state:Option<PathBuf>,resume:Option<PathBuf>,adapter:Option<PathBuf>,
    max_new:usize,sample:bool,temperature:f32,top_k:usize,top_p:f32,
    wmma:Option<&'static str>,
}
fn parse(mut argv:VecDeque<OsString>)->AResult<Options> {
    let mut o=Options{model:PathBuf::new(),train:None,image:None,output:None,tokenizer:None,prompt:String::new(),prompt_ids:None,
        batch:1,seq:512,epochs:1,device:None,hyper:CpuAdamWHyperParams{lr:5e-5,beta1:0.9,beta2:0.999,eps:1e-8,weight_decay:0.01},
        rank:8,alpha:None,seed:0,stream:false,state:None,resume:None,adapter:None,max_new:128,sample:false,temperature:0.7,top_k:50,top_p:0.9,wmma:None};
    while let Some(key)=argv.pop_front(){let key=key.to_str().context("non-UTF8 argument")?;
        match key {"--stream-parameters"=>{o.stream=true;continue;},"--do-sample"=>{o.sample=true;continue;},"--no-do-sample"=>{o.sample=false;continue;},"--amp"=>{o.wmma=Some("mixed");continue;},"--no-amp"=>{o.wmma=Some("off");continue;},_=>{}}
        let value=argv.pop_front().with_context(||format!("missing value for {key}"))?;let s=value.to_str().context("non-UTF8 value")?;
        match key {
            "--model-path"=>o.model=PathBuf::from(&value),"--train"=>o.train=Some(PathBuf::from(&value)),"--image"=>o.image=Some(PathBuf::from(&value)),
            "--out-dir"=>o.output=Some(PathBuf::from(&value)),"--tokenizer-path"=>o.tokenizer=Some(PathBuf::from(&value)),"--prompt"=>o.prompt=s.to_owned(),
            "--prompt-ids"=>o.prompt_ids=Some(s.split(',').map(str::parse).collect::<Result<_,_>>()?),
            "--batch-size"=>o.batch=s.parse()?,"--max-length"=>o.seq=s.parse()?,"--epochs"=>o.epochs=s.parse()?,"--device-index"=>o.device=Some(s.parse()?),
            "--lr"=>o.hyper.lr=s.parse()?,"--beta1"=>o.hyper.beta1=s.parse()?,"--beta2"=>o.hyper.beta2=s.parse()?,"--eps"=>o.hyper.eps=s.parse()?,"--weight-decay"=>o.hyper.weight_decay=s.parse()?,
            "--lora-rank"=>o.rank=s.parse()?,"--lora-alpha"=>o.alpha=Some(s.parse()?),"--seed"=>o.seed=s.parse()?,"--native-state"=>o.state=Some(PathBuf::from(&value)),"--resume"=>o.resume=Some(PathBuf::from(&value)),"--adapter-path"=>o.adapter=Some(PathBuf::from(&value)),
            "--max-new-tokens"=>o.max_new=s.parse()?,"--temperature"=>o.temperature=s.parse()?,"--top-k"=>o.top_k=s.parse()?,"--top-p"=>o.top_p=s.parse()?,
            "--wmma-mode"|"--wmma_mode"=>o.wmma=Some(parse_transformer_wmma_mode(s).map_err(anyhow::Error::msg)?),
            _=>anyhow::bail!("unsupported vision argument {key}; use --help"),
        }
    }
    anyhow::ensure!(!o.model.as_os_str().is_empty(),"--model-path is required");
    anyhow::ensure!(o.batch>0&&o.seq>0&&o.seq<=65_535&&o.epochs>0&&o.rank>0&&o.rank<=65_535&&o.max_new>0,"batch size, sequence length, epochs, rank and token limit must be positive; max-length and rank must be <=65535");
    o.hyper.validate()?;
    anyhow::ensure!(o.alpha.is_none_or(|a|a.is_finite()&&a>0.),"LoRA alpha must be finite and positive");
    anyhow::ensure!(o.temperature.is_finite()&&o.temperature>0.&&o.top_p.is_finite()&&o.top_p>0.&&o.top_p<=1.,"invalid sampling temperature/top-p");
    anyhow::ensure!(usize::from(o.state.is_some())+usize::from(o.resume.is_some())+usize::from(o.adapter.is_some())<=1,"choose only one native-state, resume or adapter-path");
    Ok(o)
}
pub(super) fn run(mode:&str,argv:VecDeque<OsString>)->Result<u8,String> {
    if argv.iter().any(|a|a=="--help"||a=="-h") {
        println!("Native vision commands:\n  vision-train --model-path PACKAGE --train DATA.jsonl --out-dir NEW_DIR [--batch-size 1 --max-length 512 --epochs 1 --lr 5e-5]\n  vision-generate --model-path PACKAGE --image IMAGE.png --prompt TEXT [--max-new-tokens 128 --max-length 512]\n  vision-info --model-path PACKAGE\n\nSupports hierarchos_image_text (including BART/T5/Switch), native LLaVA, Gemma3, Gemma4 and DeepSeek V4.1. Images must match the fixed native processor dimensions; no implicit resize/crop. JSONL uses image + prompt/completion or explicit input_ids/targets/loss_weights. Paths are relative to the JSONL file.\nShared/native full-parameter image training uses the existing GPU AdamW pipeline. Native V4.1 trains image parameters plus language LoRA; --lora-rank, --adapter-path, --resume, --native-state and --stream-parameters configure that path. Its output contains native image/language export and optimizer checkpoint. Other backends export a complete image model package.\nTraining precision: --wmma-mode off|mixed|full-fp16 (or --amp/--no-amp). Mixed uses FP16 operands and FP32 accumulation for eligible Transformer/vision matrix operations; other operations, master weights and AdamW remain FP32. Shared CLIP/LLaVA GPT2/Llama image transactions use joint GPU loss scaling and global optimizer skipping. Other families retain unscaled FP32 gradients. Full FP16 is restricted to the qualified shared CLIP64/Llama64 package and adapter/driver, with reductions above 64 retained in FP32; use precision-info. Native DeepSeek V4.1 accelerates eligible native vision projections; its language LoRA and short vision operations retain FP32. Requested and actual WMMA dispatches are recorded in training_report.json.\nOptional: --tokenizer-path, --device-index, --beta1, --beta2, --eps, --weight-decay, --seed. Vision generation supports --do-sample/--no-do-sample, --temperature, --top-k, --top-p and --out-dir. It generates text from an image; text-to-image uses image-generate.");return Ok(0);
    }
    execute(mode,parse(argv).map_err(|e|format!("{e:#}"))?).map(|_|0).map_err(|e|format!("native vision failed: {e:#}"))
}
fn tokenizer(p:&Package,o:&Options)->AResult<Option<Tokenizer>> {
    let path=o.tokenizer.clone().unwrap_or_else(||p.tokenizer_source());
    let file=if path.is_dir(){path.join("tokenizer.json")}else{path};
    if !file.is_file(){anyhow::ensure!(o.tokenizer.is_none(),"selected tokenizer file does not exist");return Ok(None);}
    let t=Tokenizer::from_file(&file).map_err(|e|anyhow::anyhow!("load tokenizer {}: {e}",file.display()))?;
    anyhow::ensure!(t.get_vocab(true).values().all(|&id|(id as usize)<p.vocab),"tokenizer IDs exceed image model vocabulary");Ok(Some(t))
}
fn execute(mode:&str,o:Options)->AResult<()> {
    // This command runs in its own CLI process. Set the policy before both
    // device creation and TransformerKernels construction read it.
    if let Some(wmma) = o.wmma { env::set_var("HIERARCHOS_WMMA_MODE", wmma); }
    let _full_oracle_width=crate::precision::FullOracleWidth::enter().map_err(anyhow::Error::msg)?;
    let p=Package::open(&o.model)?;
    if mode=="vision-info" {println!("{}",serde_json::to_string_pretty(&json!({"backend":format!("{:?}",p.kind),"encoder_decoder":p.seq2seq,"vocab_size":p.vocab,"image_token_id":p.image_token,"vision_config":p.config["vision_config"],"image_generation":false}))?);return Ok(());}
    let t=tokenizer(&p,&o)?;
    if mode=="vision-train" {train(&p,t.as_ref(),&o)}else{generate(&p,t.as_ref(),&o)}
}
fn train(p:&Package,t:Option<&Tokenizer>,o:&Options)->AResult<()> {
    let dataset=o.train.as_ref().context("vision-train requires --train JSONL")?;
    let output=o.output.as_ref().context("vision-train requires --out-dir")?;
    anyhow::ensure!(!output.exists()||(output.is_dir()&&fs::read_dir(output)?.next().is_none()),"output directory must be new or empty");
    anyhow::ensure!(o.state.is_none(),"training from native exported weights is not optimizer resume; use --resume with the saved checkpoint");
    let base=dataset.parent().unwrap_or_else(||Path::new("."));
    let first=BufReader::new(File::open(dataset)?).lines().find_map(|line|match line {Ok(s)if s.trim().is_empty()=>None,x=>Some(x)}).context("dataset is empty")??;
    let first=Sample::from_record(&serde_json::from_str(&first)?,base,p,t,o.seq)?;
    let shape=(first.picture.width,first.picture.height);
    let phases_before=hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts();
    let counts_before=hierarchos_vulkan::wmma_linear::linear_wmma_dispatch_counts();
    let accumulation_before=hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts();
    let reduction_before=hierarchos_vulkan::wmma_linear::full_fp16_reduction_fallbacks();
    let mut runtime=Runtime::new(p,&first.picture,o)?;
    let skipped_before=runtime.loss_scaler_state()?.map_or(0,|s|s.skipped_steps);
    fs::create_dir_all(output)?;
    let mut step=0usize;let mut loss=0.;let mut seen=0usize;
    for epoch in 0..o.epochs {
        let mut samples=Vec::with_capacity(o.batch);
        for (line_no,line) in BufReader::new(File::open(dataset)?).lines().enumerate() {
            let line=line?;if line.trim().is_empty(){continue;}
            let v:Value=serde_json::from_str(&line).with_context(||format!("dataset line {}",line_no+1))?;
            let sample=Sample::from_record(&v,base,p,t,o.seq).with_context(||format!("dataset line {}",line_no+1))?;
            anyhow::ensure!((sample.picture.width,sample.picture.height)==shape,"all images in one fixed-grid training run must have the same dimensions");
            samples.push(sample);seen+=1;
            if samples.len()==o.batch {loss=runtime.train(p,&samples,o)?;step+=1;samples.clear();eprintln!("{}",json!({"epoch":epoch+1,"step":step,"loss":loss,"examples":seen}));}
        }
        if !samples.is_empty(){loss=runtime.train(p,&samples,o)?;step+=1;eprintln!("{}",json!({"epoch":epoch+1,"step":step,"loss":loss,"examples":seen}));}
    }
    anyhow::ensure!(step>0&&loss.is_finite(),"no finite supervised updates were produced");
    let model=output.join(if p.kind==Kind::V41{"native"}else{"model"});runtime.export(p,&model)?;
    if p.kind==Kind::V41 {runtime.checkpoint(&output.join("checkpoint"))?;}
    if let Some(source)=o.tokenizer.as_ref().cloned().or_else(||t.map(|_|p.tokenizer_source())) {
        super::install_local_tokenizer_assets(&source,&model).map_err(anyhow::Error::msg)?;
    }
    let phases=hierarchos_vulkan::wmma_linear::linear_wmma_phase_counts();
    let counts=hierarchos_vulkan::wmma_linear::linear_wmma_dispatch_counts();
    let accumulation=hierarchos_vulkan::wmma_linear::linear_wmma_precision_counts();
    let precision=json!({
        "requested":hierarchos_vulkan::wmma::WmmaMode::from_env()?.label(),
        "cooperative_dispatches":counts.cooperative-counts_before.cooperative,
        "shape_fallbacks":counts.shape_fallbacks-counts_before.shape_fallbacks,
        "forward":phases[0]-phases_before[0],"weight_grad":phases[1]-phases_before[1],"input_grad":phases[2]-phases_before[2],
        "mixed_fp32_accumulation":accumulation[0]-accumulation_before[0],"full_fp16_accumulation":accumulation[1]-accumulation_before[1],
        "full_fp16_max_reduction":hierarchos_vulkan::wmma_linear::FULL_FP16_MAX_REDUCTION,
        "cpu_arithmetic_vector_width":hierarchos_vulkan::aten_vector_width(),
        "full_fp16_reduction_fallbacks":hierarchos_vulkan::wmma_linear::full_fp16_reduction_fallbacks()-reduction_before,
        "master_weights":"fp32","optimizer_state":"fp32",
        "automatic_loss_scaling":runtime.loss_scaler_state()?.is_some(),"loss_scaler":runtime.loss_scaler_state()?
    });
    eprintln!("WMMA evidence: {precision}");
    let skipped=runtime.loss_scaler_state()?.map_or(0,|s|s.skipped_steps)-skipped_before;
    anyhow::ensure!(skipped<=step as u64,"optimizer skip count exceeds attempted updates");
    fs::write(output.join("training_report.json"),serde_json::to_vec_pretty(&json!({"backend":"native-vulkan","kind":format!("{:?}",p.kind),"steps":step,"attempted_optimizer_windows":step,"optimizer_updates":step as u64-skipped,"examples":seen,"final_loss":loss,"model":model,"base_model":p.root,"image_width":shape.0,"image_height":shape.1,"batch_size":o.batch,"seq_len":o.seq,"optimizer_resume":p.kind==Kind::V41,"precision":precision}))?)?;
    println!("Vision training complete: {}",model.display());Ok(())
}
fn generate(p:&Package,t:Option<&Tokenizer>,o:&Options)->AResult<()> {
    anyhow::ensure!(o.batch==1,"vision generation accepts one image/request");
    anyhow::ensure!(o.resume.is_none(),"generation uses --native-state exported weights, not a training-resume checkpoint");
    let picture=Picture::open(o.image.as_ref().context("vision-generate requires --image")?)?;
    let mut input=if let Some(ids)=&o.prompt_ids {ids.clone()}else{anyhow::ensure!(!o.prompt.trim().is_empty(),"vision-generate requires --prompt or --prompt-ids");prompt_ids(p,&picture,t,&o.prompt)?};
    anyhow::ensure!(!input.is_empty()&&input.iter().all(|&id|(id as usize)<p.vocab),"invalid prompt token IDs");
    let mut decoder=if p.seq2seq{vec![p.decoder_start()?]}else{Vec::new()};
    anyhow::ensure!(input.len()<=o.seq&&(if p.seq2seq{decoder.len()}else{input.len()})+o.max_new<=o.seq,"prompt/image plus requested tokens exceed --max-length; increase context or lower token limit");
    let mask=vec![1.;input.len()];image_layout(p,&picture,&input,&mask)?;
    if let Some(out)=&o.output {anyhow::ensure!(!out.exists(),"output directory must be new");}
    let mut runtime=Runtime::new(p,&picture,o)?;
    let mut sampler=Sampler::new(SamplingConfig{temperature:o.temperature,top_k:o.top_k,top_p:o.top_p,repetition_penalty:1.,seed:o.seed});
    let mut result=Vec::new();
    for _ in 0..o.max_new {
        let visible=if p.seq2seq{decoder.len()}else{input.len()};
        let mut b=Batch{ids:input.clone(),mask:vec![1.;input.len()],decoder:decoder.clone(),decoder_mask:vec![1.;decoder.len()],targets:vec![0;o.seq],weights:vec![0.;o.seq],rgb:picture.rgb.clone()};
        b.ids.resize(o.seq,p.pad());b.mask.resize(o.seq,0.);if p.seq2seq{b.decoder.resize(o.seq,p.pad());b.decoder_mask.resize(o.seq,0.);}
        let all=runtime.forward(p,&picture,&b,o.seq)?;
        anyhow::ensure!(all.len()==o.seq*p.vocab,"vision backend returned an unexpected logit shape");
        let mut logits=all[(visible-1)*p.vocab..visible*p.vocab].to_vec();
        anyhow::ensure!(logits.iter().all(|v|v.is_finite()),"vision backend returned nonfinite logits");
        logits[p.image_token as usize]=f32::NEG_INFINITY;
        let token=if o.sample{sampler.sample(&logits,&result)}else{logits.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(&a.0))).context("empty vocabulary")?.0 as u32};
        if p.eos()==Some(token){break;}
        result.push(token);if p.seq2seq{decoder.push(token);}else{input.push(token);}
        if let Some(t)=t {eprintln!("{}",t.decode(&result,true).map_err(|e|anyhow::anyhow!("decode response: {e}"))?);}
    }
    let text=t.map(|t|t.decode(&result,true).map_err(|e|anyhow::anyhow!("decode response: {e}"))).transpose()?;
    let report=json!({"mode":"image_to_text","token_ids":result,"text":text,"model":p.root,"native_state":o.state});
    if let Some(out)=&o.output {fs::create_dir_all(out)?;fs::write(out.join("generation_report.json"),serde_json::to_vec_pretty(&report)?)?;}
    if let Some(text)=text{println!("{text}");}else{println!("{}",serde_json::to_string_pretty(&report)?);}Ok(())
}

#[cfg(test)]
mod tests;

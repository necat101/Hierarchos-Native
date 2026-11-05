use super::*;

pub(super) struct Sample {
    pub ids:Vec<u32>, pub mask:Vec<f32>, pub decoder:Vec<u32>, pub decoder_mask:Vec<f32>,
    pub targets:Vec<u32>, pub weights:Vec<f32>, pub picture:Picture,
}
pub(super) struct Batch {
    pub ids:Vec<u32>,pub mask:Vec<f32>,pub decoder:Vec<u32>,pub decoder_mask:Vec<f32>,
    pub targets:Vec<u32>,pub weights:Vec<f32>,pub rgb:Vec<u8>,
}
fn ids(v:&Value,name:&str)->anyhow::Result<Vec<u32>> {
    v[name].as_array().with_context(||format!("{name} must be a token array"))?.iter()
        .map(|n|Ok(u32::try_from(n.as_u64().with_context(||format!("{name} must contain unsigned token IDs"))?)?)).collect()
}
fn numbers(v:&Value,name:&str,len:usize,default:f32)->anyhow::Result<Vec<f32>> {
    if v[name].is_null(){return Ok(vec![default;len]);}
    let a=v[name].as_array().with_context(||format!("{name} must be an array"))?;
    anyhow::ensure!(a.len()==len,"{name} has the wrong length");
    a.iter().map(|x|{let n=x.as_f64().with_context(||format!("invalid {name}"))? as f32;anyhow::ensure!(n.is_finite(),"{name} must be finite");Ok(n)}).collect()
}
pub(super) fn encode(tokenizer:Option<&Tokenizer>,text:&str,special:bool)->anyhow::Result<Vec<u32>> {
    let t=tokenizer.context("raw text requires tokenizer.json; select a tokenizer or use pretokenized JSONL")?;
    Ok(t.encode(text,special).map_err(|e|anyhow::anyhow!("tokenization failed: {e}"))?.get_ids().to_vec())
}
pub(super) fn prompt_ids(p:&Package,picture:&Picture,tokenizer:Option<&Tokenizer>,text:&str)->anyhow::Result<Vec<u32>> {
    let count=p.image_types(picture.width,picture.height)?.len();
    let mut out=Vec::new();
    if let Some((before,after))=text.split_once("<image>") {
        anyhow::ensure!(!after.contains("<image>"),"exactly one image per example is supported");
        out.extend(encode(tokenizer,before,true)?);out.extend(std::iter::repeat_n(p.image_token,count));out.extend(encode(tokenizer,after,false)?);
    }else{
        if let Some(bos)=p.text["bos_token_id"].as_u64(){out.push(u32::try_from(bos)?);}
        out.extend(std::iter::repeat_n(p.image_token,count));out.extend(encode(tokenizer,text,false)?);
    }
    Ok(out)
}
pub(super) fn image_layout(p:&Package,picture:&Picture,ids:&[u32],mask:&[f32])->anyhow::Result<(Vec<i32>,usize)> {
    let image_types=p.image_types(picture.width,picture.height)?;
    let indices:Vec<_>=ids.iter().enumerate().filter_map(|(i,&id)|(id==p.image_token).then_some(i)).collect();
    anyhow::ensure!(indices.len()==image_types.len(),"expected {} image-placeholder tokens, found {}",image_types.len(),indices.len());
    anyhow::ensure!(indices.iter().all(|&i|mask[i]==1.0),"image tokens must be visible in attention_mask");
    let start=indices[0];
    let mut types=vec![-1;ids.len()];
    if p.kind==Kind::V41 {anyhow::ensure!(indices.iter().enumerate().all(|(j,&i)|i==start+j),"native V4.1 requires one contiguous complete image span");}
    for (&i,&t) in indices.iter().zip(&image_types){types[i]=t;}
    Ok((types,start))
}
impl Sample {
    pub fn from_record(v:&Value,base:&Path,p:&Package,tokenizer:Option<&Tokenizer>,seq:usize)->anyhow::Result<Self> {
        let path=v["image"].as_str().context("each record needs an image path")?;
        let path=Path::new(path);let picture=Picture::open(&if path.is_absolute(){path.to_owned()}else{base.join(path)})?;
        let explicit=v["input_ids"].is_array();
        anyhow::ensure!(explicit||(v["attention_mask"].is_null()&&v["decoder_input_ids"].is_null()),"custom masks and decoder inputs require pretokenized records");
        let mut input=if explicit {ids(v,"input_ids")?}else{prompt_ids(p,&picture,tokenizer,v["prompt"].as_str().context("record needs prompt or input_ids")?)?};
        let mut mask=numbers(v,"attention_mask",input.len(),1.)?;
        let (mut decoder,mut decoder_mask,mut targets,mut weights);
        if explicit {
            decoder=if p.seq2seq{ids(v,"decoder_input_ids")?}else{Vec::new()};
            decoder_mask=numbers(v,"decoder_attention_mask",decoder.len(),1.)?;
            targets=ids(v,"targets")?;
            let n=if p.seq2seq{decoder.len()}else{input.len()};
            anyhow::ensure!(targets.len()==n,"targets must align with the prediction rows; causal targets are already shifted by one token");
            weights=numbers(v,"loss_weights",n,1.)?;
        }else{
            let mut completion=encode(tokenizer,v["completion"].as_str().context("training record needs completion")?,false)?;
            if let Some(eos)=p.eos(){completion.push(eos);}
            anyhow::ensure!(!completion.is_empty(),"completion has no tokens");
            if p.seq2seq {
                decoder=vec![p.decoder_start()?];decoder.extend(completion.iter().take(completion.len()-1));
                decoder_mask=vec![1.;decoder.len()];targets=completion;weights=vec![1.;targets.len()];
            }else{
                let prefix=input.len();input.extend(completion);mask=vec![1.;input.len()];
                targets=input.iter().skip(1).copied().chain(std::iter::once(0)).collect();
                weights=(0..input.len()).map(|i|if i+1>=prefix&&i+1<input.len()&&input[i]!=p.image_token&&targets[i]!=p.image_token{1.}else{0.}).collect();
                decoder=Vec::new();decoder_mask=Vec::new();
            }
        }
        anyhow::ensure!(!input.is_empty()&&input.len()<=seq&&decoder.len()<=seq,"example exceeds --max-length={seq}; shorten text or use a larger context; image spans are never truncated");
        anyhow::ensure!(input.iter().chain(&decoder).all(|&id|(id as usize)<p.vocab),"input token outside vocabulary");
        anyhow::ensure!(mask.iter().chain(&decoder_mask).all(|&x|x==0.||x==1.),"attention masks must be binary");
        let visible=if p.seq2seq{&decoder_mask}else{&mask};
        anyhow::ensure!(weights.iter().all(|&w|w>=0.)&&weights.iter().any(|&w|w>0.),"record needs nonnegative loss weights and at least one supervised token");
        for i in 0..weights.len() {
            if weights[i]>0. {anyhow::ensure!((targets[i] as usize)<p.vocab&&visible[i]==1.&&(p.kind==Kind::V41||targets[i]!=p.image_token),"supervised labels must be valid visible text targets, never image placeholder labels");}
            else if targets[i] as usize>=p.vocab {targets[i]=0;}
        }
        image_layout(p,&picture,&input,&mask)?;
        input.resize(seq,p.pad());mask.resize(seq,0.);targets.resize(seq,0);weights.resize(seq,0.);
        if p.seq2seq {decoder.resize(seq,p.pad());decoder_mask.resize(seq,0.);}
        Ok(Self{ids:input,mask,decoder,decoder_mask,targets,weights,picture})
    }
}
impl Batch {
    pub fn from_samples(samples:&[Sample],batch:usize)->Self {
        let mut b=Self{ids:Vec::new(),mask:Vec::new(),decoder:Vec::new(),decoder_mask:Vec::new(),targets:Vec::new(),weights:Vec::new(),rgb:Vec::new()};
        for i in 0..batch {let s=&samples[i.min(samples.len()-1)];b.ids.extend(&s.ids);b.mask.extend(&s.mask);b.decoder.extend(&s.decoder);b.decoder_mask.extend(&s.decoder_mask);b.targets.extend(&s.targets);b.rgb.extend(&s.picture.rgb);
            if i<samples.len(){b.weights.extend(&s.weights);}else{b.weights.extend(std::iter::repeat_n(0.,s.weights.len()));}
        }b
    }
}

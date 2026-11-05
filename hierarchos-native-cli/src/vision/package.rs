use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind { Shared, Llava, Gemma3, Gemma4, V41 }

pub(super) struct Package {
    pub root: PathBuf,
    pub kind: Kind,
    pub config: Value,
    pub text: Value,
    pub vocab: usize,
    pub image_token: u32,
    pub seq2seq: bool,
}

pub(super) fn read_json(path: impl AsRef<Path>) -> anyhow::Result<Value> {
    let path=path.as_ref();
    serde_json::from_slice(&fs::read(path).with_context(||format!("read {}",path.display()))?)
        .with_context(||format!("parse {}",path.display()))
}
pub(super) fn positive(v:&Value,key:&str)->anyhow::Result<usize> {
    let n=usize::try_from(v[key].as_u64().with_context(||format!("missing positive {key}"))?)?;
    anyhow::ensure!(n>0,"{key} must be positive"); Ok(n)
}
impl Package {
    pub fn open(root:&Path)->anyhow::Result<Self> {
        let mut config=read_json(root.join("config.json"))?;
        let model=config["model_type"].as_str().context("package has no model_type")?;
        let kind=match model {
            "hierarchos_image_text"=>Kind::Shared,"llava"=>Kind::Llava,
            "gemma3"=>Kind::Gemma3,"gemma4"=>Kind::Gemma4,"deepseek_v41"=>Kind::V41,
            _=>anyhow::bail!("{model} is a text package. Select a native vision package or a hierarchos_image_text package containing text/ and image/."),
        };
        let seq2seq=kind==Kind::Shared && config["text_mode"].as_str()==Some("encoder_decoder");
        let text=if kind==Kind::Shared {read_json(root.join("text/config.json"))?}
            else {config.get("text_config").filter(|v|v.is_object()).context("vision package has no text_config")?.clone()};
        if kind==Kind::Shared {config=read_json(root.join("image/config.json"))?;}
        anyhow::ensure!(config["vision_config"].is_object(),"package has no native vision tower");
        let vocab=positive(&text,"vocab_size")?;anyhow::ensure!(vocab>1,"image/text vocabulary must contain text tokens");
        let key=if matches!(kind,Kind::V41|Kind::Gemma4){"image_token_id"}else{"image_token_index"};
        let image_token=u32::try_from(config[key].as_u64().with_context(||format!("missing {key}"))?)?;
        anyhow::ensure!((image_token as usize)<vocab,"image token is outside vocabulary");
        Ok(Self{root:root.to_owned(),kind,config,text,vocab,image_token,seq2seq})
    }
    pub fn tokenizer_source(&self)->PathBuf {
        if self.root.join("tokenizer.json").is_file(){self.root.clone()}
        else if self.kind==Kind::Shared {self.root.join("text")}else{self.root.clone()}
    }
    pub fn image_types(&self,width:usize,height:usize)->anyhow::Result<Vec<i32>> {
        let v=&self.config["vision_config"];
        let patch=positive(v,"patch_size")?;
        anyhow::ensure!(width>0&&height>0&&width%patch==0&&height%patch==0,"image dimensions must be positive multiples of patch_size={patch}; resize explicitly before use");
        if self.kind==Kind::V41 {
            let down=positive(v,"downsample_ratio")?;
            let h=(height/patch).div_ceil(down);let w=(width/patch).div_ceil(down);
            let mut types=vec![0];for _ in 0..h {types.extend(std::iter::repeat_n(1,w));types.push(2);}types.push(3);return Ok(types);
        }
        anyhow::ensure!(width==height,"this image package requires a square image");
        let count=match self.kind {
            Kind::Gemma3=>positive(&self.config,"mm_tokens_per_image")?,
            Kind::Gemma4=> {let pool=positive(v,"pooling_kernel_size")?;anyhow::ensure!((width/patch)%pool==0,"image patch grid must be divisible by pooling_kernel_size");(width/patch/pool).pow(2)},
            _=> {(width/patch).pow(2)+usize::from(self.config["vision_feature_select_strategy"].as_str()==Some("full"))},
        };
        if self.kind!=Kind::Gemma4 {anyhow::ensure!(width==positive(v,"image_size")?,"image dimensions must match vision_config.image_size; the native fixed-size processor does not resize implicitly");}
        Ok(vec![1;count])
    }
    pub fn eos(&self)->Option<u32> {
        self.text["eos_token_id"].as_u64().or_else(||self.text["eos_token_id"].as_array().and_then(|v|v.first()).and_then(Value::as_u64)).and_then(|n|u32::try_from(n).ok())
    }
    pub fn pad(&self)->u32 {
        self.text["pad_token_id"].as_u64().filter(|&id|id<(self.vocab as u64)&&id!=self.image_token as u64).map(|id|id as u32).unwrap_or_else(||if self.image_token==0{1}else{0})
    }
    pub fn decoder_start(&self)->anyhow::Result<u32> {
        let n=self.text["decoder_start_token_id"].as_u64().or_else(||self.text["bos_token_id"].as_u64()).context("encoder-decoder package requires decoder_start_token_id")?;
        anyhow::ensure!(n<(self.vocab as u64),"decoder start is outside vocabulary");Ok(n as u32)
    }
}

pub(super) struct Picture {pub rgb:Vec<u8>,pub width:usize,pub height:usize}
impl Picture {
    pub fn open(path:&Path)->anyhow::Result<Self> {
        let image=image::ImageReader::open(path).with_context(||format!("open image {}",path.display()))?.with_guessed_format()?.decode().with_context(||format!("decode image {}",path.display()))?.into_rgb8();
        Ok(Self{width:image.width() as usize,height:image.height() as usize,rgb:image.into_raw()})
    }
    pub fn patches(&self,patch:usize)->Vec<f32> {
        let mut values=Vec::with_capacity(self.rgb.len());
        for y in 0..self.height/patch {for x in 0..self.width/patch {for c in 0..3 {for py in 0..patch {for px in 0..patch {
            let v=(self.rgb[((y*patch+py)*self.width+x*patch+px)*3+c] as f32/255.0-0.5)/0.5;
            let bits=v.to_bits();values.push(f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000));
        }}}}}
        values
    }
}

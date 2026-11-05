//! Text-to-image through the native Vulkan stable-diffusion.cpp companion.
//! The image model and its numerical contract are separate from text parity.
use super::*;
use anyhow::{Context,Result as AResult};

struct Options {
    model:PathBuf,output:PathBuf,engine:Option<PathBuf>,vae:Option<PathBuf>,prompt:String,negative:String,
    width:u32,height:u32,steps:usize,cfg:f32,seed:i64,device:usize,
}
fn parse(mut args:VecDeque<OsString>)->AResult<Options> {
    let mut o=Options{model:PathBuf::new(),output:PathBuf::new(),engine:None,vae:None,prompt:String::new(),negative:String::new(),width:512,height:512,steps:20,cfg:7.,seed:42,device:0};
    while let Some(key)=args.pop_front(){let key=key.to_str().context("non-UTF8 image argument")?;
        let value=args.pop_front().with_context(||format!("missing value for {key}"))?;let s=value.to_str().context("non-UTF8 image value")?;
        match key {
            "--model-path"=>o.model=PathBuf::from(value),"--output"=>o.output=PathBuf::from(value),"--engine"=>o.engine=Some(PathBuf::from(value)),"--vae"=>o.vae=Some(PathBuf::from(value)),
            "--prompt"=>o.prompt=s.to_owned(),"--negative-prompt"=>o.negative=s.to_owned(),"--width"=>o.width=s.parse()?,"--height"=>o.height=s.parse()?,
            "--steps"=>o.steps=s.parse()?,"--cfg-scale"=>o.cfg=s.parse()?,"--seed"=>o.seed=s.parse()?,"--device-index"=>o.device=s.parse()?,
            _=>anyhow::bail!("unsupported image-generation argument {key}; use --help"),
        }
    }
    anyhow::ensure!(o.model.is_file(),"--model-path must select a Stable Diffusion single-file checkpoint (.safetensors/.gguf/.ckpt)");
    anyhow::ensure!(!o.prompt.trim().is_empty(),"--prompt must not be empty");
    anyhow::ensure!(o.output.extension().is_some_and(|e|e.eq_ignore_ascii_case("png")),"--output must name a new PNG file");
    anyhow::ensure!(!o.output.exists(),"output PNG already exists; choose a new filename");
    anyhow::ensure!([o.width,o.height].iter().all(|&n|(64..=4096).contains(&n)&&n%64==0),"image dimensions must be multiples of 64 between 64 and 4096");
    anyhow::ensure!((1..=1000).contains(&o.steps)&&o.cfg.is_finite()&&o.cfg>0.&&o.seed>= -1,"invalid steps, CFG scale or seed");
    if let Some(vae)=&o.vae {anyhow::ensure!(vae.is_file(),"VAE checkpoint does not exist");}
    Ok(o)
}
fn engine_path(explicit:Option<&Path>)->PathBuf {
    if let Some(path)=explicit {return path.to_owned();}
    if let Some(path)=env::var_os("HIERARCHOS_SD_CLI"){return PathBuf::from(path);}
    let name=if cfg!(windows){"sd-cli.exe"}else{"sd-cli"};
    if let Ok(exe)=env::current_exe(){if let Some(dir)=exe.parent(){
        for path in [dir.join(name),dir.join("image-engine").join(name),dir.join("bin").join(name)]{if path.is_file(){return path;}}
        for parent in dir.ancestors().take(6){let path=parent.join("target/image-engine").join(name);if path.is_file(){return path;}}
    }}
    if let Ok(cwd)=env::current_dir(){let path=cwd.join("target/image-engine").join(name);if path.is_file(){return path;}}
    PathBuf::from(name)
}
fn hidden_command(exe:&Path)->Command {
    let mut cmd=Command::new(exe);
    #[cfg(windows)] {use std::os::windows::process::CommandExt;cmd.creation_flags(0x08000000);}
    cmd
}
fn hash_file(path:&Path)->AResult<String> {
    let mut file=File::open(path)?;let mut hash=Sha256::new();let mut buffer=[0u8;64*1024];
    loop{let n=file.read(&mut buffer)?;if n==0{break;}hash.update(&buffer[..n]);}Ok(format!("{:x}",hash.finalize()))
}
fn engine_args(o:&Options,output:&Path)->Vec<OsString> {
    let mut a:Vec<OsString>=vec!["-m".into(),o.model.as_os_str().into(),"-p".into(),o.prompt.clone().into(),"-o".into(),output.as_os_str().into(),
        "--width".into(),o.width.to_string().into(),"--height".into(),o.height.to_string().into(),"--steps".into(),o.steps.to_string().into(),
        "--cfg-scale".into(),o.cfg.to_string().into(),"--seed".into(),o.seed.to_string().into(),"--batch-count".into(),"1".into(),
        "--backend".into(),format!("Vulkan{}",o.device).into(),"--params-backend".into(),format!("Vulkan{}",o.device).into(),
        "--auto-fit".into(),"off".into(),"--sampling-method".into(),"euler".into(),"--vae-tiling".into(),"--preview".into(),"none".into()];
    if !o.negative.is_empty(){a.extend(["--negative-prompt".into(),o.negative.clone().into()]);}
    if let Some(vae)=&o.vae{a.extend(["--vae".into(),vae.as_os_str().into()]);}a
}
struct Staging(PathBuf);
impl Drop for Staging {fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}

pub(super) fn run(mut argv:VecDeque<OsString>,info:bool)->Result<u8,String> {
    if argv.iter().any(|a|a=="--help"||a=="-h") {
        println!("Native Vulkan text-to-image:\n  image-generate --model-path IMAGE_MODEL --prompt TEXT --output NEW.png [--engine SD_CLI]\n  image-engine-info [--engine SD_CLI]\n\nUses the native stable-diffusion.cpp sd-cli companion; no Python runtime. Single-file Stable Diffusion checkpoints (SD1.x/SD2.x/SDXL, optionally --vae PATH). The image model is separate from the 44 qualified language backbones. Install the pinned Vulkan engine with hierarchos-native-cli/install-image-engine.ps1, colocate sd-cli and its DLLs, or set HIERARCHOS_SD_CLI.\nOptions: --width 512 --height 512 --steps 20 --cfg-scale 7 --seed 42 --negative-prompt TEXT --device-index 0. Generation explicitly selects Vulkan for compute and parameters and disables automatic CPU/disk offload. Dimensions must be multiples of 64; choose settings that fit the selected GPU. Existing output files are preserved. A verified PNG and .generation.json receipt are published only after successful generation.");return Ok(0);
    }
    let result=(||->AResult<()> {
        if info {
            let explicit=if argv.is_empty(){None}else{anyhow::ensure!(argv.pop_front().as_deref()==Some(std::ffi::OsStr::new("--engine"))&&argv.len()==1,"image-engine-info accepts only --engine PATH");Some(PathBuf::from(argv.pop_front().unwrap()))};
            let engine=engine_path(explicit.as_deref());let status=hidden_command(&engine).arg("--list-devices").status().with_context(||format!("launch image engine {}; run install-image-engine.ps1 or select --engine",engine.display()))?;
            anyhow::ensure!(status.success(),"image engine could not enumerate devices: {status}");return Ok(());
        }
        let o=parse(argv)?;let engine=engine_path(o.engine.as_deref());
        let help=hidden_command(&engine).arg("--help").output().with_context(||format!("launch image engine {}; run install-image-engine.ps1 or select --engine",engine.display()))?;
        let version=format!("{}{}",String::from_utf8_lossy(&help.stdout),String::from_utf8_lossy(&help.stderr));
        anyhow::ensure!(help.status.success()&&["--backend","--params-backend","--auto-fit"].iter().all(|flag|version.contains(flag)),"selected sd-cli lacks explicit Vulkan backend selection; install the pinned supported version");
        let parent=o.output.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or_else(||Path::new("."));fs::create_dir_all(parent)?;
        let receipt_path=o.output.with_extension("generation.json");anyhow::ensure!(!receipt_path.exists(),"generation receipt already exists; choose a new filename");
        let nonce=SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();let staging=parent.join(format!(".hierarchos-image-{}-{nonce}",std::process::id()));
        fs::create_dir(&staging)?;let staging=Staging(staging);let temporary=staging.0.join("result.png");
        let args=engine_args(&o,&temporary);
        eprintln!("Text-to-image: native Vulkan device {}, {}x{}, {} Euler steps",o.device,o.width,o.height,o.steps);
        let started=Instant::now();
        let status=hidden_command(&engine).args(&args).stdin(Stdio::null()).status()?;
        anyhow::ensure!(status.success(),"image engine exited with {status}; no output was published");
        let candidates=[temporary,staging.0.join("result_1.png")];let outputs:Vec<_>=candidates.iter().filter(|p|p.is_file()).collect();
        anyhow::ensure!(outputs.len()==1,"engine returned success without exactly one expected output PNG");let generated=outputs[0];
        let bytes=fs::read(generated)?;anyhow::ensure!(image::guess_format(&bytes)?==image::ImageFormat::Png,"engine output is not a PNG");
        let decoded=image::load_from_memory(&bytes)?;anyhow::ensure!(decoded.width()==o.width&&decoded.height()==o.height,"engine PNG dimensions differ from the request");
        let record=json!({"schema":1,"kind":"text_to_image","backend":"stable-diffusion.cpp-vulkan","device":format!("Vulkan{}",o.device),"automatic_offload":false,
            "engine":engine,"engine_version":version.lines().next(),"engine_sha256":if engine.is_file(){Some(hash_file(&engine)?)}else{None},"model":o.model,"model_sha256":hash_file(&o.model)?,"vae":o.vae,"vae_sha256":o.vae.as_ref().map(|p|hash_file(p)).transpose()?,
            "prompt":o.prompt,"negative_prompt":o.negative,"width":o.width,"height":o.height,"steps":o.steps,"cfg_scale":o.cfg,"seed":o.seed,
            "sampling_method":"euler","output":o.output,"png_sha256":format!("{:x}",Sha256::digest(&bytes)),"elapsed_seconds":started.elapsed().as_secs_f64(),"language_logit_parity_applies":false});
        let report_bytes=serde_json::to_vec_pretty(&record)?;
        let mut final_image=fs::OpenOptions::new().write(true).create_new(true).open(&o.output)?;
        if let Err(e)=final_image.write_all(&bytes).and_then(|_|final_image.sync_all()){drop(final_image);let _=fs::remove_file(&o.output);return Err(e.into());}drop(final_image);
        let mut report=match fs::OpenOptions::new().write(true).create_new(true).open(&receipt_path){Ok(file)=>file,Err(e)=>{let _=fs::remove_file(&o.output);return Err(e.into());}};
        if let Err(e)=report.write_all(&report_bytes).and_then(|_|report.sync_all()){drop(report);let _=fs::remove_file(&o.output);let _=fs::remove_file(&receipt_path);return Err(e.into());}
        println!("Generated image: {}",o.output.display());println!("Generation receipt: {}",receipt_path.display());Ok(())
    })();
    result.map(|_|0).map_err(|e|format!("text-to-image failed: {e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_engine_arguments_preserve_prompts_and_force_vulkan() {
        let o=Options{model:PathBuf::from("model with spaces.gguf"),output:PathBuf::from("out.png"),engine:None,vae:None,prompt:"a cat; $(literal)".into(),negative:"text, blur".into(),width:512,height:512,steps:3,cfg:1.,seed:42,device:1};
        let args=engine_args(&o,Path::new("output with spaces.png"));
        let find=|flag:&str|args.windows(2).find(|a|a[0]==flag).unwrap()[1].clone();
        assert_eq!(find("-p"),OsString::from("a cat; $(literal)"));assert_eq!(find("--backend"),OsString::from("Vulkan1"));assert_eq!(find("--params-backend"),OsString::from("Vulkan1"));assert_eq!(find("--auto-fit"),OsString::from("off"));
        assert_eq!(find("-o"),OsString::from("output with spaces.png"));assert!(!args.iter().any(|a|a=="--clip-on-cpu"));
    }
}

//! Native checkpoint arithmetic, also used by the external PEFT oracle.
use anyhow::{bail, Context, Result};
use hierarchos_vulkan::{merge_peft_lora_safetensors, unmerge_peft_lora_safetensors};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let operation = args.next().context("expected merge or unmerge")?;
    let source = PathBuf::from(args.next().context("expected source SafeTensors path")?);
    let adapter = PathBuf::from(args.next().context("expected adapter directory")?);
    let destination = PathBuf::from(args.next().context("expected destination SafeTensors path")?);
    if args.next().is_some() { bail!("unexpected checkpoint argument"); }
    let report = match operation.to_str() {
        Some("merge") => merge_peft_lora_safetensors(&source, &adapter, &destination)?,
        Some("unmerge") => unmerge_peft_lora_safetensors(&source, &adapter, &destination)?,
        _ => bail!("expected merge or unmerge"),
    };
    println!("{report:?}");
    Ok(())
}

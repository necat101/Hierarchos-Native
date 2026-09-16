use std::path::PathBuf;

use anyhow::{Context, Result};
use hierarchos_vulkan::read_adamw_optimizer_state;
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    step: u32,
    tensor_count: usize,
    slot_names: Vec<String>,
    slot_steps: Vec<u32>,
    slot_decay_classes: Vec<Option<String>>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut optimizer = None;
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--optimizer" => optimizer = args.next().map(PathBuf::from),
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let optimizer = optimizer.context("missing --optimizer PATH")?;
    let state = read_adamw_optimizer_state(&optimizer)
        .with_context(|| format!("reading optimizer checkpoint {}", optimizer.display()))?;
    let tensor_count = state.slots.len();
    let slot_names = state.slots.iter().map(|slot| slot.name.clone()).collect();
    let slot_steps = state.slots.iter().map(|slot| slot.step).collect();
    let slot_decay_classes = state
        .slots
        .iter()
        .map(|slot| {
            slot.decay_class
                .map(|decay_class| decay_class.checkpoint_label().to_string())
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&Output {
            step: state.step,
            tensor_count,
            slot_names,
            slot_steps,
            slot_decay_classes,
        })?
    );
    Ok(())
}

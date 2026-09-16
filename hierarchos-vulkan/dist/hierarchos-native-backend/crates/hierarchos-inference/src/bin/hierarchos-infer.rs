use std::{env, path::PathBuf};

use hierarchos_inference::HierarchosModel;
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    architecture_revision: String,
    architecture_contract_sha256: Option<String>,
    tokens: Vec<u32>,
    state_position: usize,
    logits: Vec<Vec<f32>>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir: Option<PathBuf> = None;
    let mut token_text: Option<String> = None;
    let mut load_state: Option<PathBuf> = None;
    let mut save_state: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--tokens" => token_text = args.next(),
            "--load-state" => load_state = args.next().map(PathBuf::from),
            "--save-state" => save_state = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                println!(
                    "Usage: hierarchos-infer --model MODEL_DIR --tokens 1,2,3 \
                     [--load-state runtime_state.json] [--save-state runtime_state.json]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let model_dir = model_dir.ok_or("--model is required")?;
    let token_text = token_text.ok_or("--tokens is required")?;
    let tokens: Vec<u32> = token_text
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<u32>())
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("--tokens must contain at least one token id".into());
    }

    let model = HierarchosModel::load(model_dir)?;
    let architecture_revision = model.config().architecture_revision.clone();
    let architecture_contract_sha256 = model.config().architecture_contract_sha256.clone();
    let mut state = match load_state {
        Some(path) => model.load_runtime_state_json(path)?,
        None => model.new_state(),
    };
    let logits = model.prefill(&tokens, &mut state)?;
    if let Some(path) = save_state {
        model.save_runtime_state_json(&state, path)?;
    }
    let state_position = state.position();
    println!(
        "{}",
        serde_json::to_string(&Output {
            architecture_revision,
            architecture_contract_sha256,
            tokens,
            state_position,
            logits,
        })?
    );
    Ok(())
}

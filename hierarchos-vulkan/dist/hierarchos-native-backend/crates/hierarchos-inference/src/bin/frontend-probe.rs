use std::{fs, path::PathBuf};

use hierarchos_inference::HierarchosModel;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    token_ids: Vec<u32>,
    token_residual: Option<Vec<f32>>,
    gated_ltm_values: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    enc: Vec<f32>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_dir: Option<PathBuf> = None;
    let mut case_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--case" => case_path = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let model = HierarchosModel::load(model_dir.ok_or("--model is required")?)?;
    let case: Case = serde_json::from_slice(&fs::read(case_path.ok_or("--case is required")?)?)?;
    if case.token_ids.is_empty() {
        return Err("case token_ids must not be empty".into());
    }
    let context_dim = model.config().context_dim;
    let ltm_dim = model.config().ltm_topk * model.config().ltm_val_dim;
    let expected_residual = case.token_ids.len() * context_dim;
    if let Some(residual) = &case.token_residual {
        if residual.len() != expected_residual {
            return Err(format!(
                "token_residual has {} values; expected {expected_residual}",
                residual.len()
            )
            .into());
        }
    }
    let expected_ltm = case.token_ids.len() * ltm_dim;
    if case.gated_ltm_values.len() != expected_ltm {
        return Err(format!(
            "gated_ltm_values has {} values; expected {expected_ltm}",
            case.gated_ltm_values.len()
        )
        .into());
    }

    let mut enc = Vec::with_capacity(case.token_ids.len() * context_dim);
    for (row, &token) in case.token_ids.iter().enumerate() {
        let residual = case
            .token_residual
            .as_deref()
            .map(|values| &values[row * context_dim..(row + 1) * context_dim]);
        let ltm = &case.gated_ltm_values[row * ltm_dim..(row + 1) * ltm_dim];
        enc.extend(model.project_token_frontend(token, residual, ltm)?);
    }
    println!("{}", serde_json::to_string(&Output { enc })?);
    Ok(())
}

//! One coherent-v9 RWKV cell step on the CPU runtime, for cross-backend checks.
//!
//! The Vulkan `rwkv-packed-cell-step` runner consumes the same case fields and
//! reports the same `(output, packed_new_state)` pair, so a generated package and
//! case can be compared on both runtimes without either side re-implementing the
//! other's graph. Rows are independent: each one starts from its own packed state
//! slice, exactly as the batched Vulkan dispatch does.

use std::{fs, path::PathBuf};

use hierarchos_inference::rwkv_cell_step;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    batch: usize,
    width: usize,
    head_size: usize,
    #[serde(default)]
    state_clamp: f32,
    #[serde(default)]
    key_clamp: f32,
    #[serde(default)]
    deepembed_clamp: f32,
    x: Vec<f32>,
    deepembed: Vec<f32>,
    packed_state: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    batch: usize,
    width: usize,
    head_size: usize,
    state_size: usize,
    output: Vec<f32>,
    packed_new_state: Vec<f32>,
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
    let mut prefix = String::from("h_rnn");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("missing --model-dir value")?,
                ))
            }
            "--case" => case_path = Some(PathBuf::from(args.next().ok_or("missing --case value")?)),
            "--cell-prefix" => prefix = args.next().ok_or("missing --cell-prefix value")?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let model_dir = model_dir.ok_or("rwkv cell probe requires --model-dir")?;
    let case_path = case_path.ok_or("rwkv cell probe requires --case")?;
    let case: Case = serde_json::from_slice(&fs::read(case_path)?)?;

    if case.batch == 0 || case.width == 0 || case.head_size == 0 {
        return Err("rwkv cell probe requires positive batch/width/head_size".into());
    }
    let state_size = 4 + case.head_size;
    let vector_len = case.batch * case.width;
    let state_len = vector_len * state_size;
    if case.x.len() != vector_len || case.packed_state.len() != state_len {
        return Err(format!(
            "case geometry mismatch: x={} (expected {vector_len}), packed_state={} (expected {state_len})",
            case.x.len(),
            case.packed_state.len()
        )
        .into());
    }
    if case.deepembed.len() != vector_len * 4 {
        return Err(format!(
            "case deepembed has {} values; expected {}",
            case.deepembed.len(),
            vector_len * 4
        )
        .into());
    }

    let mut output = Vec::with_capacity(vector_len);
    let mut packed_new_state = Vec::with_capacity(state_len);
    for row in 0..case.batch {
        let x = &case.x[row * case.width..(row + 1) * case.width];
        let deepembed = &case.deepembed[row * case.width * 4..(row + 1) * case.width * 4];
        let packed =
            &case.packed_state[row * case.width * state_size..(row + 1) * case.width * state_size];
        let (row_output, row_state) = rwkv_cell_step(
            &model_dir,
            &prefix,
            case.width,
            case.head_size,
            case.state_clamp,
            case.key_clamp,
            case.deepembed_clamp,
            x,
            deepembed,
            packed,
        )?;
        output.extend_from_slice(&row_output);
        packed_new_state.extend_from_slice(&row_state);
    }

    println!(
        "{}",
        serde_json::to_string(&Output {
            batch: case.batch,
            width: case.width,
            head_size: case.head_size,
            state_size,
            output,
            packed_new_state,
        })?
    );
    Ok(())
}

//! Report the physical device's cooperative-matrix contract as negotiated by
//! the requested `--wmma-mode`.
//!
//! This is the receipt behind the "capability checks" requirement: it prints
//! what the driver actually advertises (extension, feature, every configuration
//! row, subgroup width) and what the logical device actually enables. It never
//! infers support from the adapter name and never claims WMMA for a device whose
//! feature is off.

use std::process::ExitCode;

use anyhow::{Context, Result};
use hierarchos_vulkan::wmma::{WmmaMode, WMMA_MODE_ENV};
use hierarchos_vulkan::VulkanDevice;

fn usage() -> &'static str {
    "hierarchos-vulkan-wmma-probe [--wmma-mode off|mixed|full-fp16] [--json]\n\n\
     With no --wmma-mode the probe uses HIERARCHOS_WMMA_MODE, or `off` when unset.\n\
     `off` still reports the advertised configurations; it simply does not enable\n\
     the cooperative-matrix feature, so it can be used as a non-invasive probe."
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("wmma-probe: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut mode = WmmaMode::from_env()?;
    let mut json = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wmma-mode" => {
                let value = args.next().context("--wmma-mode requires a value")?;
                mode = WmmaMode::parse(&value)?;
            }
            "--json" => json = true,
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => anyhow::bail!("unsupported argument {other:?}\n\n{}", usage()),
        }
    }

    let device = VulkanDevice::new_for_wmma(mode)
        .with_context(|| format!("creating a Vulkan device with --wmma-mode {}", mode.label()))?;
    let capabilities = device.wmma_capabilities();

    if json {
        let tile = |value: Option<hierarchos_vulkan::WmmaTile>| match value {
            Some(tile) => format!(
                "{{\"m\":{},\"n\":{},\"k\":{},\"a\":\"{}\",\"b\":\"{}\",\"c\":\"{}\",\"subgroup_scope\":{},\"saturating\":{}}}",
                tile.m,
                tile.n,
                tile.k,
                if tile.a_is_f16 { "f16" } else { "other" },
                if tile.b_is_f16 { "f16" } else { "other" },
                if tile.c_is_f16 {
                    "f16"
                } else if tile.c_is_f32 {
                    "f32"
                } else {
                    "other"
                },
                tile.subgroup_scope,
                tile.saturating
            ),
            None => "null".to_owned(),
        };
        println!(
            concat!(
                "{{\n",
                "  \"device\": {:?},\n",
                "  \"physical_device_index\": {},\n",
                "  \"mode\": {:?},\n",
                "  \"extension_exposed\": {},\n",
                "  \"feature_supported\": {},\n",
                "  \"memory_model_supported\": {},\n",
                "  \"subgroup_size_control_supported\": {},\n",
                "  \"storage_16bit\": {},\n",
                "  \"shader_float16\": {},\n",
                "  \"driver_subgroup_size\": {},\n",
                "  \"enabled\": {},\n",
                "  \"configuration_count\": {},\n",
                "  \"mixed_tile\": {},\n",
                "  \"full_fp16_tile\": {},\n",
                "  \"mixed_supported\": {},\n",
                "  \"full_fp16_supported\": {}\n",
                "}}"
            ),
            device.name(),
            device.physical_device_index(),
            mode.label(),
            capabilities.extension_exposed,
            capabilities.feature_supported,
            capabilities.memory_model_supported,
            capabilities.subgroup_size_control_supported,
            capabilities.storage_16bit,
            capabilities.shader_float16,
            capabilities.subgroup_size,
            capabilities.enabled,
            capabilities.configuration_count,
            tile(capabilities.mixed_tile),
            tile(capabilities.full_fp16_tile),
            capabilities.mixed_supported(),
            capabilities.full_fp16_supported(),
        );
        return Ok(());
    }

    println!("device: {}", device.name());
    println!("physical device index: {}", device.physical_device_index());
    println!("requested mode: {} ({})", mode.label(), mode.description());
    println!("environment: {WMMA_MODE_ENV}={mode}", mode = mode.label());
    print!("{}", capabilities.diagnostics());
    println!(
        "supported: mixed={} full-fp16={}",
        capabilities.mixed_supported(),
        capabilities.full_fp16_supported()
    );
    Ok(())
}

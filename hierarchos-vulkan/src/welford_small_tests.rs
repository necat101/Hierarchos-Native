use crate::{
    vulkan::{ComputeBatch, ComputeKernel},
    GpuBuffer, VulkanDevice,
};
use anyhow::Result;
use std::time::Instant;
#[test]
fn small_welford_matches_general_path_across_all_short_widths() -> Result<()> {
    let Ok(device) = VulkanDevice::new() else {
        return Ok(());
    };
    if device.vendor_id() != 0x1002 {
        return Ok(());
    }
    let original = ComputeKernel::new(
        &device,
        include_bytes!("../shaders/transformer_layer_norm_welford_forward.spv"),
        6,
        12,
    )?;
    let small = ComputeKernel::new(
        &device,
        include_bytes!("../shaders/transformer_layer_norm_welford_small_forward.spv"),
        6,
        12,
    )?;
    let mut seed = 1337u32;
    let mut random = |n| {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / 16777216. - 0.5
            })
            .collect::<Vec<_>>()
    };
    let mut results = Vec::new();
    for dim in 1usize..=128 {
        for rows in [1usize, 6, 17, 64] {
            let x = GpuBuffer::from_f32(&device, &random(rows * dim))?;
            let w = GpuBuffer::from_f32(&device, &random(dim))?;
            let b = GpuBuffer::from_f32(&device, &random(dim))?;
            let mut expected = Vec::new();
            let mut timings = Vec::new();
            for (kernel, rows_per_group) in [(&original, 64), (&small, 8)] {
                let y = GpuBuffer::zeros_f32(&device, rows * dim)?;
                let mean = GpuBuffer::zeros_f32(&device, rows)?;
                let rstd = GpuBuffer::zeros_f32(&device, rows)?;
                let push = [(rows as u32), (dim as u32), 1e-5f32.to_bits()];
                let mut commands = ComputeBatch::new(&device)?;
                kernel.record_dispatch(
                    &mut commands,
                    &[&x, &w, &b, &y, &mean, &rstd],
                    bytemuck::cast_slice(&push),
                    [rows.div_ceil(rows_per_group) as u32, 1, 1],
                )?;
                commands.submit()?;
                let actual = [
                    y.read_f32(rows * dim)?,
                    mean.read_f32(rows)?,
                    rstd.read_f32(rows)?,
                ]
                .concat()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
                if expected.is_empty() {
                    expected = actual;
                } else {
                    assert_eq!(expected, actual, "dim {dim} rows {rows}");
                }
                if [16, 32, 64, 128].contains(&dim) && rows == 6 {
                    let mut samples = Vec::new();
                    for _ in 0..30 {
                        let start = Instant::now();
                        let mut commands = ComputeBatch::new(&device)?;
                        for _ in 0..100 {
                            kernel.record_dispatch(
                                &mut commands,
                                &[&x, &w, &b, &y, &mean, &rstd],
                                bytemuck::cast_slice(&push),
                                [1, 1, 1],
                            )?;
                        }
                        commands.submit()?;
                        samples.push(start.elapsed().as_secs_f64() * 10.);
                    }
                    samples.sort_by(f64::total_cmp);
                    timings.push(samples[15]);
                }
            }
            if !timings.is_empty() {
                results.push(serde_json::json!({"dim":dim,"rows":rows,"original_ms":timings[0],"small_ms":timings[1],"ratio":timings[1]/timings[0]}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"result":"pass","geometries":512,"timings":results})
    );
    Ok(())
}

#[test]
fn small_welford_admission_is_bounded_to_the_measured_adapter_and_width() {
    let device = "AMD Radeon Graphics [pci=1002:15bf;driver=00800184]";
    for width in [1, 8, 16, 64, 127, 128] {
        assert!(super::small_welford_is_qualified(device, width));
    }
    for width in [0, 129, 256, 768] {
        assert!(!super::small_welford_is_qualified(device, width));
    }
    for other in [
        "Intel(R) HD Graphics 520",
        "AMD Radeon Graphics [pci=1002:15bf;driver=00800185]",
    ] {
        assert!(!super::small_welford_is_qualified(other, 16));
    }
}

use super::*;

#[test]
#[ignore = "requires the explicit serial-control benchmark runner"]
fn rdna3_gpt2_paired_gpu_benchmark() -> Result<()> {
    let device = VulkanDevice::new()?;
    anyhow::ensure!(crate::vendor::rdna3_avx512_variants(
        &device,
        aten_vector_width()
    ));
    let root = std::path::PathBuf::from(std::env::var("HIERARCHOS_RDNA3_GPT2_BENCH")?);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("benchmark-config.json"))?)?;
    let requested_lanes = config["forward_lanes"].as_u64().unwrap() as u32;
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/rdna3_gpt2_attention.json"
    ))?;
    let mut cases = Vec::new();
    for seq in [8u32, 16] {
        let lanes = if requested_lanes == 0 {
            if seq == 8 {
                8
            } else {
                1
            }
        } else {
            requested_lanes
        };
        let c = corpus["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["batch"] == 2 && c["seq"] == seq && c["heads"] == 4 && c["padded"] == false)
            .unwrap();
        let values = |key: &str| -> Vec<f32> {
            c[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(v.as_i64().unwrap() as u32))
                .collect()
        };
        let push = [2, seq, 64, 64, 4, 4, 0, 0.25f32.to_bits(), 1];
        let len = (2 * seq * 64) as usize;
        for backward in [false, true] {
            let phase = if backward { "backward" } else { "forward" };
            let mut data = vec![
                values("q"),
                values("k"),
                values("v"),
                values("mask"),
                values("mask"),
            ];
            if backward {
                data.extend([
                    values("dy"),
                    values("o"),
                    vec![0.; len],
                    vec![0.; len],
                    vec![0.; len],
                ]);
            } else {
                data.push(vec![0.; len]);
            }
            let buffers = data
                .iter()
                .map(|v| GpuBuffer::from_f32(&device, v))
                .collect::<Result<Vec<_>>>()?;
            let bindings = buffers.iter().collect::<Vec<_>>();
            let serial = std::fs::read(root.join(format!("{phase}.spv")))?;
            let packed: &[u8] = if backward {
                include_bytes!("../../shaders/image_text_gpt2_backward_avx512_amd_rdna3.spv")
            } else {
                include_bytes!("../../shaders/image_text_gpt2_avx512_amd_rdna3.spv")
            };
            let candidate = if backward || (requested_lanes == 0 && lanes == 1) {
                None
            } else {
                Some(std::fs::read(root.join("forward-packed.spv"))?)
            };
            let packed = candidate.as_deref().unwrap_or(packed);
            let kernels = [
                vulkan::ComputeKernel::new(&device, &serial, buffers.len(), 36)?,
                vulkan::ComputeKernel::new(&device, packed, buffers.len(), 36)?,
            ];
            let grids = if backward {
                [[4, 2, 1], [4, 2, 1]]
            } else {
                [[2 * seq * 4, 1, 1], [(2 * seq * 4).div_ceil(lanes), 1, 1]]
            };
            let mut reference: Option<Vec<Vec<u32>>> = None;
            let mut samples = Vec::new();
            for round in 0..9 {
                for index in [0usize, 1, 1, 0] {
                    let before = device.submission_arena_stats()?;
                    let ms = device.time_compute_batch_ms(|commands| {
                        for _ in 0..32 {
                            kernels[index].record_dispatch(
                                commands,
                                &bindings,
                                bytemuck::cast_slice(&push),
                                grids[index],
                            )?;
                        }
                        Ok(())
                    })?;
                    let after = device.submission_arena_stats()?;
                    let timer = if after.kernel_timestamp_profile_samples
                        == before.kernel_timestamp_profile_samples + 1
                    {
                        "gpu_timestamp"
                    } else {
                        "submit_wait"
                    };
                    let outputs = buffers[if backward { 7 } else { 5 }..]
                        .iter()
                        .map(|b| {
                            Ok(b.read_f32(len)?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>())
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if let Some(expected) = &reference {
                        assert_eq!(&outputs, expected, "serial scheduling changed outputs");
                    } else {
                        reference = Some(outputs);
                    }
                    if round > 0 {
                        samples.push(serde_json::json!({"round":round-1,"mode":if index==0 {"serial"} else {"packed"},"batch_ms":ms,"timer":timer}));
                    }
                }
            }
            cases.push(
                serde_json::json!({"sequence":seq,"phase":phase,"samples":samples,
                "selected_lanes":if backward {64} else {lanes}}),
            );
        }
    }
    std::fs::write(
        root.join("timings.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"isolated persistent-buffer kernels; timestamp queries with submit/wait fallback; not production throughput",
        "rounds":8,"order":"ABBA","dispatches_per_batch":32,"outputs_word_identical":true,"cases":cases,
        "binary":std::env::current_exe()?,"vendor_id":device.vendor_id(),"device_id":device.device_id(),
        "driver_version":device.driver_version()}))?,
    )?;
    Ok(())
}

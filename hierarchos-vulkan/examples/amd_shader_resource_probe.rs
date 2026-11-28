//! Query `VK_AMD_shader_info` for the committed attention-backward modules.
//!
//! The Phase XVIII brief asks whether the specialized SPIR-V really changes the
//! backend's resource allocation, not merely its source form. `spirv-dis` can
//! answer the indexing question but not the residency one; this probe asks the
//! IHV compiler behind the driver for the statistics it keeps for a built
//! pipeline: VGPRs, SGPRs, scratch, LDS and the required workgroup size.
//!
//! It builds a compute pipeline from each module with the descriptor and push
//! constant shape the production kernel uses, so the answer is comparable across
//! arms. It is an inspection tool, not part of any training path.
//!
//! ```text
//! cargo run --release --example amd_shader_resource_probe -- \
//!   shaders/transformer_attention_backward_query_rdna3.spv \
//!   shaders/transformer_attention_backward_query_rdna3_dqunroll.spv
//! ```

use std::ffi::{CStr, c_void};

use ash::vk;

const STORAGE_BINDINGS: usize = 9;
const PUSH_CONSTANT_BYTES: u32 = 120;

/// `VkShaderResourceUsageAMD`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShaderResourceUsageAmd {
    num_used_vgprs: u32,
    num_used_sgprs: u32,
    lds_size_per_local_work_group: u32,
    lds_usage_size_in_bytes: usize,
    scratch_mem_usage_in_bytes: usize,
}

/// `VkShaderStatisticsInfoAMD`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShaderStatisticsInfoAmd {
    shader_stage_mask: vk::ShaderStageFlags,
    resource_usage: ShaderResourceUsageAmd,
    num_physical_vgprs: u32,
    num_physical_sgprs: u32,
    num_available_vgprs: u32,
    num_available_sgprs: u32,
    compute_work_group_size: [u32; 3],
    num_physical_devices: u32,
}

type GetShaderInfoAmd = unsafe extern "system" fn(
    vk::Device,
    vk::Pipeline,
    vk::ShaderStageFlags,
    u32,
    *mut usize,
    *mut c_void,
) -> vk::Result;

const SHADER_INFO_TYPE_STATISTICS: u32 = 0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: amd_shader_resource_probe <shader.spv>...");
        std::process::exit(2);
    }

    unsafe {
        let entry = ash::Entry::load()?;
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default(), None)?;
        let physical_devices = instance.enumerate_physical_devices()?;
        let mut chosen = None;
        for physical_device in physical_devices {
            let extensions = instance.enumerate_device_extension_properties(physical_device)?;
            let has_info = extensions
                .iter()
                .any(|extension| {
                    CStr::from_ptr(extension.extension_name.as_ptr()) == vk::AMD_SHADER_INFO_NAME
                });
            let queues = instance.get_physical_device_queue_family_properties(physical_device);
            let compute_queue = queues
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE));
            if has_info && compute_queue.is_some() {
                chosen = Some((physical_device, compute_queue.unwrap() as u32));
                break;
            }
        }
        let (physical_device, queue_family_index) =
            chosen.ok_or("no compute device exposing VK_AMD_shader_info")?;
        let properties = instance.get_physical_device_properties(physical_device);
        let device_name = CStr::from_ptr(properties.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        let queue_priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];
        let extension_names = [vk::AMD_SHADER_INFO_NAME.as_ptr()];
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extension_names),
            None,
        )?;

        let proc_name = b"vkGetShaderInfoAMD\0";
        // `vkGetInstanceProcAddr` lives on the entry table; the instance handle
        // (not the physical device) is its first argument.
        let raw = unsafe {
            entry.get_instance_proc_addr(instance.handle(), proc_name.as_ptr().cast())
        };
        if raw.is_none() {
            return Err("vkGetShaderInfoAMD is not resolvable".into());
        }
        let get_shader_info: GetShaderInfoAmd = std::mem::transmute(raw.unwrap());

        let binding_flags = [vk::DescriptorBindingFlags::empty()];
        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&binding_flags);
        let bindings = (0..STORAGE_BINDINGS)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .push_next(&mut binding_flags_info),
            None,
        )?;
        let set_layouts = [set_layout];
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_range),
            None,
        )?;

        println!(
            "probe device={device_name:?} physical_devices={} push_bytes={PUSH_CONSTANT_BYTES} \
             storage_bindings={STORAGE_BINDINGS} shader_info_type=statistics",
            properties.device_type.as_raw()
        );

        for path in paths {
            let bytes = std::fs::read(&path)?;
            let words = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let pipelines = device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
            let pipeline = match pipelines {
                Ok(mut created) => created.remove(0),
                Err((_, error)) => return Err(format!("{path}: pipeline creation failed: {error}").into()),
            };

            let mut size = 0usize;
            let status = get_shader_info(
                device.handle(),
                pipeline,
                vk::ShaderStageFlags::COMPUTE,
                SHADER_INFO_TYPE_STATISTICS,
                &mut size,
                std::ptr::null_mut(),
            );
            let prefix = std::mem::size_of::<ShaderStatisticsInfoAmd>();
            if status != vk::Result::SUCCESS || size < prefix {
                println!(
                    "{path} status={status:?} struct_bytes={size} \
                     expected={prefix} unavailable"
                );
                device.destroy_pipeline(pipeline, None);
                device.destroy_shader_module(module, None);
                continue;
            }
            // LLPC reports 72 bytes for this structure, 8 more than the
            // documented layout. Allocate exactly what the driver asked for and
            // read the documented prefix instead of insisting on an exact size.
            let driver_bytes = size;
            let mut raw = vec![0u64; size.div_ceil(8)];
            let status = get_shader_info(
                device.handle(),
                pipeline,
                vk::ShaderStageFlags::COMPUTE,
                SHADER_INFO_TYPE_STATISTICS,
                &mut size,
                raw.as_mut_ptr().cast(),
            );
            if status != vk::Result::SUCCESS {
                println!("{path} status={status:?} statistics unavailable");
            } else {
                // SAFETY: the buffer is at least `prefix` bytes and the driver
                // reports the documented prefix first.
                let stats = unsafe { &*(raw.as_ptr().cast::<ShaderStatisticsInfoAmd>()) };
                let usage = stats.resource_usage;
                println!(
                    "{path} driver_struct_bytes={driver_bytes} vgpr={} sgpr={} \
                     physical_vgpr={} physical_sgpr={} available_vgpr={} \
                     available_sgpr={} scratch_bytes={} lds_bytes={} \
                     lds_per_workgroup={} workgroup={}x{}x{} physical_devices={} \
                     stage_mask=0x{:x}",
                    usage.num_used_vgprs,
                    usage.num_used_sgprs,
                    stats.num_physical_vgprs,
                    stats.num_physical_sgprs,
                    stats.num_available_vgprs,
                    stats.num_available_sgprs,
                    usage.scratch_mem_usage_in_bytes,
                    usage.lds_usage_size_in_bytes,
                    usage.lds_size_per_local_work_group,
                    stats.compute_work_group_size[0],
                    stats.compute_work_group_size[1],
                    stats.compute_work_group_size[2],
                    stats.num_physical_devices,
                    stats.shader_stage_mask.as_raw(),
                );
            }
            device.destroy_pipeline(pipeline, None);
            device.destroy_shader_module(module, None);
        }
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

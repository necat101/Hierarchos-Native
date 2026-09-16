use anyhow::{bail, Result};
use hierarchos_vulkan::VulkanDevice;
use serde_json::json;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let probe_external_self = match args.as_slice() {
        [] => false,
        [flag] if flag == "--probe-external-self" => true,
        _ => bail!("usage: hierarchos-vulkan-devices [--probe-external-self]"),
    };
    let devices = VulkanDevice::enumerate_compute_devices()?;
    let output = devices
        .into_iter()
        .map(|device| {
            let external_candidate = device.external_buffer.platform_bidirectional_candidate()
                && device.external_semaphore.platform_bidirectional_candidate();
            let mut value = json!({
                "index": device.index,
                "name": device.name,
                "device_type": device.device_type,
                "compute_queue_family_index": device.compute_queue_family_index,
                "device_uuid": device.device_uuid,
                "driver_uuid": device.driver_uuid,
                "device_group": device.device_group.map(|group| json!({
                    "group_index": group.group_index,
                    "physical_device_count": group.physical_device_count,
                    "subset_allocation": group.subset_allocation,
                })),
                "external_buffer": {
                    "opaque_win32_extension_exposed": device.external_buffer.opaque_win32_extension_exposed,
                    "opaque_win32_exportable": device.external_buffer.opaque_win32_exportable,
                    "opaque_win32_importable": device.external_buffer.opaque_win32_importable,
                    "opaque_fd_extension_exposed": device.external_buffer.opaque_fd_extension_exposed,
                    "opaque_fd_exportable": device.external_buffer.opaque_fd_exportable,
                    "opaque_fd_importable": device.external_buffer.opaque_fd_importable,
                    "platform_bidirectional_candidate": device.external_buffer.platform_bidirectional_candidate(),
                    "platform_handle": device.external_buffer.platform_handle_name(),
                },
                "external_semaphore": {
                    "opaque_win32_extension_exposed": device.external_semaphore.opaque_win32_extension_exposed,
                    "opaque_win32_exportable": device.external_semaphore.opaque_win32_exportable,
                    "opaque_win32_importable": device.external_semaphore.opaque_win32_importable,
                    "opaque_fd_extension_exposed": device.external_semaphore.opaque_fd_extension_exposed,
                    "opaque_fd_exportable": device.external_semaphore.opaque_fd_exportable,
                    "opaque_fd_importable": device.external_semaphore.opaque_fd_importable,
                    "platform_bidirectional_candidate": device.external_semaphore.platform_bidirectional_candidate(),
                    "platform_handle": device.external_semaphore.platform_handle_name(),
                },
            });
            if probe_external_self {
                value["opaque_external_self_probe"] = if external_candidate {
                    match VulkanDevice::probe_opaque_external_transport_indices(
                        device.index,
                        device.index,
                    ) {
                        Ok(probe) => json!({
                            "ok": true,
                            "handle": probe.handle_name,
                            "payload_bytes": probe.payload_bytes,
                            "synchronized_roundtrip": probe.synchronized_roundtrip,
                        }),
                        Err(err) => json!({
                            "ok": false,
                            "error": format!("{err:#}"),
                        }),
                    }
                } else {
                    json!({
                        "ok": false,
                        "skipped": "platform external memory/semaphore capability is not bidirectional",
                    })
                };
            }
            value
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

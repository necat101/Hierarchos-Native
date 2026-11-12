use super::*;

pub(super) struct FullOracleWidth {
    previous: Option<OsString>,
    active: bool,
}
impl FullOracleWidth {
    pub fn enter() -> Result<Self, String> {
        let active = hierarchos_vulkan::wmma::WmmaMode::from_env().map_err(|e| e.to_string())?
            == hierarchos_vulkan::wmma::WmmaMode::FullFp16;
        let previous = env::var_os("HIERARCHOS_ATEN_VECTOR_WIDTH");
        if active {
            env::set_var("HIERARCHOS_ATEN_VECTOR_WIDTH", "8");
        }
        Ok(Self { previous, active })
    }
}
impl Drop for FullOracleWidth {
    fn drop(&mut self) {
        if self.active {
            if let Some(value) = &self.previous {
                env::set_var("HIERARCHOS_ATEN_VECTOR_WIDTH", value);
            } else {
                env::remove_var("HIERARCHOS_ATEN_VECTOR_WIDTH");
            }
        }
    }
}

pub(super) fn info(mut args: VecDeque<OsString>) -> Result<u8, String> {
    let mut index = None;
    while let Some(arg) = args.pop_front() {
        match arg.to_string_lossy().as_ref() {
            "--device-index" => {
                index = Some(
                    args.pop_front()
                        .ok_or("missing --device-index value")?
                        .to_string_lossy()
                        .parse::<usize>()
                        .map_err(|e| e.to_string())?,
                )
            }
            "--help" | "-h" => {
                println!("precision-info [--device-index N]: query native cooperative precision capabilities as JSON");
                return Ok(0);
            }
            other => return Err(format!("unsupported precision-info argument {other:?}")),
        }
    }
    let full = VulkanDevice::new_with_wmma(index, hierarchos_vulkan::wmma::WmmaMode::FullFp16);
    let full_error = full.as_ref().err().map(|e| format!("{e:#}"));
    let full_opened = full.is_ok();
    let device = full
        .or_else(|_| VulkanDevice::new_with_wmma(index, hierarchos_vulkan::wmma::WmmaMode::Mixed));
    let value = match device {
        Ok(d) => json!({"kind":"precision-capabilities","device_index":index.unwrap_or(0),
            "device":d.name(),"device_key":d.decision_device_name(),"vendor_id":d.vendor_id(),"device_id":d.device_id(),
            "mixed_supported":d.wmma_capabilities().mixed_supported(),
            "full_fp16_supported":full_opened&&d.wmma_capabilities().full_fp16_supported(),
            "full_fp16_qualified":full_opened&&hierarchos_vulkan::precision_policy::full_fp16_hardware_qualified(
                d.vendor_id(),d.device_id(),d.decision_device_name()),
            "full_fp16_max_reduction":hierarchos_vulkan::wmma_linear::FULL_FP16_MAX_REDUCTION,"error":full_error}),
        Err(error) => json!({"kind":"precision-capabilities","device_index":index.unwrap_or(0),
            "mixed_supported":false,"full_fp16_supported":false,"full_fp16_qualified":false,"error":format!("{error:#}")}),
    };
    println!("{}", value);
    Ok(0)
}

pub(super) fn require_full_device(device: &VulkanDevice) -> Result<(), String> {
    if !hierarchos_vulkan::precision_policy::full_fp16_hardware_qualified(
        device.vendor_id(),
        device.device_id(),
        device.decision_device_name(),
    ) {
        return Err("Full FP16 currently requires the numerically qualified RDNA3 adapter and driver; this device has no matching qualification.".into());
    }
    device
        .wmma_capabilities()
        .require(hierarchos_vulkan::wmma::WmmaMode::FullFp16)
        .map_err(|e| e.to_string())
}

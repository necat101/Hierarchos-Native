//! Gen9 FMA variants for the V4.1 CPU arithmetic contract.
use crate::{vulkan, VulkanDevice};

pub(super) fn shader<'a>(device: &VulkanDevice, native: &'a [u8]) -> &'a [u8] {
    if let Some(shader) = super::cpu_math::shader(device, native) {
        return shader;
    }
    if !vulkan::is_intel_gen9_pci_id(device.vendor_id(), device.device_id()) {
        return native;
    }
    if native == include_bytes!("../../shaders/transformer_deepseek_v41_hc_split.spv")
        && std::env::var_os("HIERARCHOS_VULKAN_GEN9_PARALLEL_HC").is_some_and(|v| v == "0")
    {
        return native;
    }
    const PAIRS: &[(&[u8], &[u8])] = &[
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_linear_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_linear_half.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_linear_half_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output_input_grad.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_grouped_output_input_grad_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_gather_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_gather_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_query_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_query_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weight_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weight_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_index_scores.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_index_scores_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_vl.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_vl_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/linear_input_grad.spv"),
            include_bytes!("../../shaders/linear_input_grad_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/linear_input_grad_lane2.spv"),
            include_bytes!("../../shaders/linear_input_grad_lane2_gen9.spv"),
        ),
    ];
    PAIRS
        .iter()
        .find_map(|&(original, gen9)| (native == original).then_some(gen9))
        .unwrap_or(native)
}

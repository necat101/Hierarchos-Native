//! Select the independently pinned AVX2 oracle at synchronous kernel construction.
//! Existing text fixtures retain their original arithmetic. The thread-local
//! guard is private, restores nesting/unwinds, cannot cross threads, and never
//! changes a constructed kernel or any other thread's model. Public native VL
//! entry points scope every construction, including dynamically staged graphs.
use crate::{vulkan, VulkanDevice};
use std::{cell::Cell, marker::PhantomData, rc::Rc};
thread_local! { static AVX2: Cell<bool> = const { Cell::new(false) }; }
pub(super) struct Scope {
    previous: bool,
    _thread: PhantomData<Rc<()>>,
}
pub(super) fn avx2() -> Scope {
    Scope {
        previous: AVX2.with(|p| p.replace(true)),
        _thread: PhantomData,
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        AVX2.with(|p| p.set(self.previous));
    }
}
pub(super) fn selected<'a>(device: &VulkanDevice, original: &'a [u8]) -> &'a [u8] {
    shader(device, original).unwrap_or(original)
}
pub(super) fn active() -> bool {
    AVX2.with(Cell::get)
}
pub(super) fn vector_width() -> u32 {
    if active() {
        8
    } else {
        crate::transformer::aten_vector_width()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_avx2_scope_preserves_host_dispatch() {
        let host_width = crate::transformer::aten_vector_width();
        assert!(!super::active());
        {
            let _outer = super::avx2();
            assert_eq!(crate::transformer::aten_vector_width(), 8);
            {
                let _inner = super::avx2();
                assert_eq!(super::vector_width(), 8);
            }
            assert!(super::active());
            let unwind = std::panic::catch_unwind(|| {
                let _inner = super::avx2();
                panic!("exercise scope restoration");
            });
            assert!(unwind.is_err());
            assert!(super::active());
        }
        assert!(!super::active());
        assert_eq!(crate::transformer::aten_vector_width(), host_width);
    }
}
pub(super) fn shader(device: &VulkanDevice, original: &[u8]) -> Option<&'static [u8]> {
    if !AVX2.with(Cell::get) {
        return None;
    }
    const PAIRS: &[(&[u8], &[u8], &[u8])] = &[
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weights_avx2.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_attention_weights_avx2_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_values_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weight_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_weight_grad_avx2.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_attention_weight_grad_avx2_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_query_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_query_grad_avx2.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_attention_query_grad_avx2_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_gather_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_attention_gather_grad_avx2.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_attention_gather_grad_avx2_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output_input_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_input_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_input_grad_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_mixes_input_grad_avx2.spv"),
            include_bytes!(
                "../../shaders/transformer_deepseek_v41_hc_mixes_input_grad_avx2_gen9.spv"
            ),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_linear.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_linear_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_linear_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_grouped_output_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_hc_split_grad_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_mixed_swiglu_grad_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_engram_gate_grad_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_vl.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_vl_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_vl_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_gate_grad_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_moe_expert_grad_avx2_gen9.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_expert_fused.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_expert_fused_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_expert_fused_avx2.spv"),
        ),
        (
            include_bytes!("../../shaders/transformer_deepseek_v41_unary_test.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_unary_test_avx2.spv"),
            include_bytes!("../../shaders/transformer_deepseek_v41_unary_test_avx2.spv"),
        ),
    ];
    let gen9 = vulkan::is_intel_gen9_pci_id(device.vendor_id(), device.device_id());
    PAIRS.iter().find_map(|&(base, avx2, avx2_gen9)| {
        (base == original).then_some(if gen9 { avx2_gen9 } else { avx2 })
    })
}

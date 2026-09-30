//! Qualification metadata is deliberately separate from native topology support.
//! Promote an entry only after independent base and common PEFT oracle gates.
use super::VulkanTransformerArchitecture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeftModuleClass { Linear, LayerNorm, LmHead, Embedding }

#[derive(Debug, Clone, Copy)]
pub struct PeftCapability {
    pub base_parity_verified: bool,
    pub validation_status: &'static str,
    pub supports_peft: bool,
    pub validated_methods: &'static [&'static str],
    pub targetable_classes: &'static [PeftModuleClass],
    pub saved_module_classes: &'static [PeftModuleClass],
    pub canonical_traversal: &'static str,
    pub replacement_bank_available: bool,
    pub modules_to_save_validated: bool,
    pub tied_weight_constraints: &'static str,
    pub exclusions: &'static str,
    pub evidence: Option<&'static str>,
}

impl VulkanTransformerArchitecture {
    /// Independent native base gate, including the additional README families.
    /// This creates PEFT follow-up work; it never asserts PEFT qualification.
    pub fn peft_base_verified(self) -> bool {
        matches!(self, Self::Gpt2 | Self::Llama | Self::Mixtral | Self::Qwen3Next
            | Self::DeepseekV4 | Self::Phi3 | Self::DeepseekV3
            | Self::KimiLinear | Self::GptOss | Self::SmolLm3 | Self::Qwen2
            | Self::Qwen35 | Self::Qwen35Moe | Self::Qwen4Exp | Self::Mistral4
            | Self::MiniMaxM3VLText | Self::Gemma4 | Self::MiniMaxM2 | Self::Gemma3 | Self::FalconH1)
    }
    /// Audited capabilities, never inferred from a loader alias or base support.
    pub fn peft_capability(self) -> PeftCapability {
        use PeftModuleClass::*;
        // Keep this list explicit even though it currently matches the native
        // base-green inventory.  PEFT promotion is a separate gate: every entry
        // below has deterministic logits/gradient/two-step-AdamW, frozen-base,
        // resume, save/reload, merge/unmerge and multi-adapter evidence at the
        // fixed <=2e-7 absolute tolerance.
        let qualified = matches!(
            self,
            Self::Gpt2
                | Self::Llama
                | Self::Mixtral
                | Self::Qwen3Next
                | Self::DeepseekV4
                | Self::Phi3
                | Self::DeepseekV3
                | Self::KimiLinear
                | Self::GptOss
                | Self::SmolLm3
                | Self::Qwen2
                | Self::Qwen35
                | Self::Qwen35Moe
                | Self::Qwen4Exp
                | Self::Mistral4
                | Self::MiniMaxM3VLText
                | Self::Gemma4
                | Self::MiniMaxM2
                | Self::Gemma3
                | Self::FalconH1
        );
        // Saved-module qualification is deliberately stricter than ordinary
        // LoRA.  Architectures with even one documented native-green surface
        // above the gradient gate remain fail-closed here.
        let modules_to_save_validated = matches!(
            self,
            Self::Gpt2
                | Self::Llama
                | Self::Mixtral
                | Self::Qwen3Next
                | Self::DeepseekV4
                | Self::Phi3
                | Self::DeepseekV3
                | Self::KimiLinear
                | Self::GptOss
                | Self::SmolLm3
                | Self::Qwen2
                | Self::Qwen35
                | Self::Qwen35Moe
                | Self::Qwen4Exp
                | Self::Mistral4
                | Self::MiniMaxM3VLText
                | Self::Gemma4
                | Self::MiniMaxM2
                | Self::Gemma3
                | Self::FalconH1
        );
        let exclusions = match self {
            _ if qualified => "multi-adapter trainable tokens, layer replication and OLoRA/PiSSA remain unsupported; replacement classes are restricted by the common validator",
            _ => "PEFT unqualified: require independent base parity <=2e-7, then register canonical modules and pass the common PEFT suite",
        };
        PeftCapability {
            base_parity_verified: self.peft_base_verified(),
            validation_status: if qualified && modules_to_save_validated { "LoRA and modules_to_save qualified for documented fixtures" }
                else if qualified { "LoRA qualified; modules_to_save remains fail-closed for at least one documented fixture" }
                else if self.peft_base_verified() { "PEFT validation pending; see complete audit matrix" }
                else { "awaiting native base gate" },
            supports_peft: qualified,
            validated_methods: if qualified { &["LoRA"] } else { &[] },
            targetable_classes: if qualified { &[Linear] } else { &[] },
            // LayerNorm includes registered RMSNorm nodes, not arbitrary
            // internal/per-head norms outside the canonical replacement graph.
            saved_module_classes: if qualified {
                &[Linear, LayerNorm, LmHead, Embedding]
            } else { &[] },
            canonical_traversal: "VulkanTransformer::peft_layer_linears (canonical HF names)",
            replacement_bank_available: self.peft_base_verified(),
            // Separate gate: availability never constitutes parity evidence.
            modules_to_save_validated,
            tied_weight_constraints: "saved lm_head clones the tied base; saved embedding preserves the canonical output head unless ensure_weight_tying requests a shared replacement",
            exclusions,
            evidence: if qualified {
                Some("PROGRESS_PEFT_AUDIT.md: 2026-09-30 fresh 32/32 all-stage local-Transformers qualification, frozen-base/resume and shader provenance")
            } else {
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VulkanTransformerArchitecture;

    #[test]
    fn documented_peft_families_have_independent_saved_module_qualification() {
        assert!(VulkanTransformerArchitecture::Gpt2.peft_capability().modules_to_save_validated);
        assert!(VulkanTransformerArchitecture::Llama.peft_capability().modules_to_save_validated);
        assert!(
            VulkanTransformerArchitecture::Qwen3Next
                .peft_capability()
                .modules_to_save_validated
        );
        for architecture in [
            VulkanTransformerArchitecture::Qwen35,
            VulkanTransformerArchitecture::Qwen35Moe,
            VulkanTransformerArchitecture::Qwen4Exp,
            VulkanTransformerArchitecture::Qwen2,
            VulkanTransformerArchitecture::MiniMaxM3VLText,
            VulkanTransformerArchitecture::MiniMaxM2,
        ] {
            assert!(
                architecture.peft_capability().modules_to_save_validated,
                "{architecture:?}"
            );
        }

        let mixtral = VulkanTransformerArchitecture::Mixtral.peft_capability();
        assert!(mixtral.supports_peft);
        assert!(mixtral.replacement_bank_available);
        assert!(mixtral.modules_to_save_validated);

        for architecture in [
            VulkanTransformerArchitecture::DeepseekV4,
            VulkanTransformerArchitecture::Phi3,
            VulkanTransformerArchitecture::DeepseekV3,
            VulkanTransformerArchitecture::KimiLinear,
            VulkanTransformerArchitecture::GptOss,
            VulkanTransformerArchitecture::SmolLm3,
            VulkanTransformerArchitecture::Qwen2,
            VulkanTransformerArchitecture::Qwen35,
            VulkanTransformerArchitecture::Qwen35Moe,
            VulkanTransformerArchitecture::Qwen4Exp,
            VulkanTransformerArchitecture::Mistral4,
            VulkanTransformerArchitecture::MiniMaxM3VLText,
            VulkanTransformerArchitecture::Gemma4,
            VulkanTransformerArchitecture::MiniMaxM2,
            VulkanTransformerArchitecture::Gemma3,
            VulkanTransformerArchitecture::FalconH1,
        ] {
            assert!(architecture.peft_capability().supports_peft, "{architecture:?}");
        }

        for architecture in [
            VulkanTransformerArchitecture::Gemma4,
            VulkanTransformerArchitecture::Gemma3,
        ] {
            let capability = architecture.peft_capability();
            assert!(capability.supports_peft, "{architecture:?}");
            assert!(capability.modules_to_save_validated, "{architecture:?}");
            assert!(capability.saved_module_classes.contains(&super::PeftModuleClass::LayerNorm));
            assert!(!capability.exclusions.contains("fail-closed"));
        }
    }

    #[test]
    fn unqualified_architectures_remain_fail_closed() {
        for architecture in [VulkanTransformerArchitecture::Bert, VulkanTransformerArchitecture::Gemma] {
            let capability = architecture.peft_capability();
            assert!(!capability.supports_peft);
            assert!(!capability.modules_to_save_validated);
            assert!(capability.validated_methods.is_empty());
            assert!(capability.saved_module_classes.is_empty());
        }
    }
}

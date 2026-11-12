//! Read-only operation tape for independent GPT-2 parity diagnostics.
use super::*;

impl VulkanTransformer {
    #[doc(hidden)]
    pub fn debug_gpt2_forward_states(&self) -> Result<BTreeMap<String, Vec<f32>>> {
        if self.config.architecture != VulkanTransformerArchitecture::Gpt2 {
            bail!("GPT-2 operation trace requires gpt2");
        }
        let rows = self.rows;
        let hidden = rows * self.config.hidden_size;
        let mut out = BTreeMap::new();
        out.insert("input".into(), self.input_hidden.read_f32(hidden)?);
        out.insert(
            "final_norm".into(),
            self.final_norm_output.read_f32(hidden)?,
        );
        out.insert(
            "logits".into(),
            self.logits.read_f32(rows * self.config.vocab_size)?,
        );
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, buffer, len) in [
                ("ln1", &layer.tape.ln1, hidden),
                ("q", &layer.tape.q, rows * layer.query_hidden_size),
                ("k", &layer.tape.k, rows * layer.key_hidden_size),
                ("v", &layer.tape.v, rows * layer.value_hidden_size),
                ("attention", &layer.tape.attention, hidden),
                ("o", &layer.tape.attention_projection, hidden),
                ("residual", &layer.tape.residual1, hidden),
                ("ln2", &layer.tape.ln2, hidden),
                ("u", &layer.tape.mlp_pre, rows * layer.c_fc.output_dim),
                (
                    "activation",
                    &layer.tape.mlp_activation,
                    rows * layer.c_fc.output_dim,
                ),
                ("d", &layer.tape.mlp_output, hidden),
                ("out", &layer.tape.output, hidden),
            ] {
                out.insert(format!("layer_{index}/{name}"), buffer.read_f32(len)?);
            }
        }
        Ok(out)
    }
}

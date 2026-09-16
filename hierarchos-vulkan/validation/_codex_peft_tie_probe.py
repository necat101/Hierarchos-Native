from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM


model = LlamaForCausalLM(
    LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        tie_word_embeddings=True,
    )
)
for modules_to_save in (["embed_tokens"], ["lm_head"], ["embed_tokens", "lm_head"]):
    candidate = LlamaForCausalLM(model.config)
    candidate.load_state_dict(model.state_dict())
    config = LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["q_proj"],
        modules_to_save=modules_to_save,
        ensure_weight_tying=True,
    )
    print("case", modules_to_save)
    try:
        peft_model = get_peft_model(candidate, config)
        runtime_config = peft_model.peft_config["default"]
        print("modules_to_tie", getattr(runtime_config, "modules_to_tie", None))
        embedding_wrapper = getattr(peft_model.base_model.model.model.embed_tokens, "modules_to_save", None)
        head_wrapper = getattr(peft_model.base_model.model.lm_head, "modules_to_save", None)
        if embedding_wrapper is not None and head_wrapper is not None:
            print("shared_weight", embedding_wrapper.default.weight is head_wrapper.default.weight)
        print("state_keys")
        for key in sorted(get_peft_model_state_dict(peft_model)):
            print(key)
    except Exception as error:
        print("error", type(error).__name__, str(error))

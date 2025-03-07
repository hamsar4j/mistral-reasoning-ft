from model.config import ModelConfig
from unsloth import FastLanguageModel


def load_model_for_inference(model_config: ModelConfig, adapter_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=model_config.load_in_4bit,
        attn_implementation=model_config.attn_implementation,
        use_cache=model_config.use_cache,
        device_map=model_config.device_map,
        dtype=model_config.dtype,
        max_seq_length=model_config.max_seq_length,
    )
    model = FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.6,
        min_p=0.95,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    # response = tokenizer.batch_decode(outputs)
    return response

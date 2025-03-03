from peft import PeftModel
from model.config import ModelConfig
from unsloth import FastLanguageModel


def load_model_for_inference(model_config: ModelConfig, adapter_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_id,
        load_in_4bit=model_config.load_in_4bit,
        attn_implementation=model_config.attn_implementation,
        use_cache=model_config.use_cache,
        device_map=model_config.device_map,
        dtype=model_config.dtype,
        max_seq_length=model_config.max_seq_length,
    )
    model = FastLanguageModel.for_inference(model)

    # load adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    # model = model.merge_and_unload()

    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=8092,
        do_sample=True,
        temperature=1.5,
        min_p=0.1,
    )
    # response = tokenizer.decode(
    #     outputs[0][inputs.shape[1] :], skip_special_tokens=False
    # )
    response = tokenizer.batch_decode(outputs)
    return response

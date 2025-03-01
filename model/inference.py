from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from model.config import ModelConfig


def load_model_for_inference(model_config: ModelConfig, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    special_tokens = {
        "additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]
    }
    tokenizer.add_special_tokens(special_tokens)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=model_config.get_quantization_config(),
        attn_implementation=model_config.attn_implementation,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # model = model.merge_and_unload()

    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt"
    ).to(model.device)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        # pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(
        outputs[0][inputs.shape[1] :], skip_special_tokens=False
    )
    return response

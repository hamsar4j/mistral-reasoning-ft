from .config import ModelConfig
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset


def setup_model(model_config: ModelConfig, lora_config: LoraConfig):

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=model_config.get_quantization_config(),
        attn_implementation=model_config.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

    # add custom special tokens
    special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
    tokenizer.add_special_tokens(special_tokens)

    # resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config.get_lora_config())

    return peft_model, tokenizer


def setup_trainer(
    model,
    train_dataset: Dataset,
    training_config: TrainingArguments,
):

    training_args = training_config.get_training_args()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    return trainer

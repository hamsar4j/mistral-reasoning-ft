from .config import ModelConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset
from unsloth import FastLanguageModel


def setup_model(model_config: ModelConfig, lora_config: LoraConfig):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_id,
        load_in_4bit=model_config.load_in_4bit,
        attn_implementation=model_config.attn_implementation,
        use_cache=model_config.use_cache,
        device_map=model_config.device_map,
        dtype=model_config.dtype,
        max_seq_length=model_config.max_seq_length,
    )

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
    )

    return peft_model, tokenizer


def setup_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    training_config: SFTConfig,
):

    training_args = training_config.get_training_args()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    return trainer

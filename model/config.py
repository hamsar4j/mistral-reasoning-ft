import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # quantization params
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # model params
    attn_implementation: str = "flash_attention_2"
    use_cache: bool = False
    device_map: str = "auto"
    torch_dtype: torch.dtype = "auto"

    def get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list = field(default_factory=lambda: ["lm_head", "embed_token"])
    task_type: str = "CAUSAL_LM"

    def get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            task_type=self.task_type,
        )


@dataclass
class TrainingConfig:
    output_dir: str = "models/mistral-7b-reasoning-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-4
    logging_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048

    def get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            save_strategy=self.save_strategy,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            max_seq_length=self.max_seq_length,
        )

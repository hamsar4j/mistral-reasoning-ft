import torch
from peft import LoraConfig
from trl import SFTConfig
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    # quantization params
    load_in_4bit: bool = True
    # model params
    attn_implementation: str = "flash_attention_2"
    use_cache: bool = False
    device_map: str = "auto"
    torch_dtype: torch.dtype = "auto"
    max_seq_length: int = 32768


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"
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
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407

    def get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            target_modules=self.target_modules,
        )


@dataclass
class TrainingConfig:
    output_dir: str = "models/mistral-7b-reasoning-lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    learning_rate: float = 1e-4
    logging_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "linear"
    save_strategy: str = "epoch"
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 32768

    def get_training_args(self) -> SFTConfig:
        return SFTConfig(
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
            max_seq_length=self.max_seq_length,
        )

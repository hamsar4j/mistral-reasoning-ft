from trl import SFTConfig
from dataclasses import dataclass, field
from unsloth import is_bfloat16_supported


@dataclass
class ModelConfig:
    model_id: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    # quantization params
    load_in_4bit: bool = True
    # model params
    attn_implementation: str = "flash_attention_2"
    use_cache: bool = False
    device_map: str = "auto"
    dtype: str = "None"
    max_seq_length: int = 2048


@dataclass
class LoRAConfig:
    r: int = 64
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


@dataclass
class TrainingConfig:
    output_dir: str = "models/mistral-7b-reasoning-lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    learning_rate: float = 5e-6
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()

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
            fp16=self.fp16,
            bf16=self.bf16,
        )

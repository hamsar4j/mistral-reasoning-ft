from data.preprocessing import process_cot_example, preprocess_dataset
from utils.utils import load_training_dataset, setup_logging
from model.config import ModelConfig, LoRAConfig, TrainingConfig
from model.trainer import setup_model, setup_trainer
import logging


def train():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting training setup")

    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()

    dataset_path = "openai/gsm8k"

    logger.info(f"Loading model: {model_config.model_id}")
    model, tokenizer = setup_model(model_config, lora_config)

    logger.info("Loading dataset")
    train_ds = load_training_dataset(dataset_path)

    logger.info("Preprocessing training data...")
    processed_train_ds = preprocess_dataset(train_ds, tokenizer, process_cot_example)
    logger.info(f"Processed Training Data Example: {processed_train_ds[0]}")

    logger.info("Starting trainer...")
    trainer = setup_trainer(model, tokenizer, processed_train_ds, training_config)

    trainer.train()
    trainer.save_model(training_config.output_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    train()

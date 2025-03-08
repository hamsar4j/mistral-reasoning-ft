import logging

from model.inference import load_model_for_inference, generate_response
from utils.utils import setup_logging
from model.config import ModelConfig


def run_inference():
    setup_logging()
    logger = logging.getLogger(__name__)

    adapter_path = "models/mistral-7b-reasoning-lora"
    model_config = ModelConfig()

    logger.info(f"Loading model from {adapter_path}")
    model, tokenizer = load_model_for_inference(model_config, adapter_path)
    logger.info("Model loaded successfully")

    prompt = "How many 'r' are there in the word 'strawberry'? Think before answering."

    response = generate_response(model, tokenizer, prompt)
    print(response)


if __name__ == "__main__":
    run_inference()

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

    prompt = "Consider the following two person game. A number of pebbles are situated on the table. Two players make their moves alternately. A move consists of taking off the table  $x$  pebbles where  $x$  is the square of any positive integer. The player who is unable to make a move loses. Prove that there are infinitely many initial situations in which the second player can win no matter how his opponent plays. show me your thinking and reasoning."

    response = generate_response(model, tokenizer, prompt)
    print(response)


if __name__ == "__main__":
    run_inference()

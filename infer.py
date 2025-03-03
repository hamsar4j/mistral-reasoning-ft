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

    prompt = "One base of a trapezoid is $100$ units longer than the other base. The segment that joins the midpoints of the legs divides the trapezoid into two regions whose areas are in the ratio $2: 3$ . Let $x$ be the length of the segment joining the legs of the trapezoid that is parallel to the bases and that divides the trapezoid into two regions of equal area. Find the greatest integer that does not exceed $x^2/100$. Show your thinking before you answer."

    response = generate_response(model, tokenizer, prompt)
    print(response)


if __name__ == "__main__":
    run_inference()

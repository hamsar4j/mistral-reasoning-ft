import re
import numpy as np


def check_response_format(predictions):
    results = {
        "has_think_section": [],
        "has_answer_section": [],
        "has_both_sections": [],
    }

    for pred in predictions:
        has_think = bool(re.search(r"\[THINK\].*?\[/THINK\]", pred, re.DOTALL))
        has_answer = bool(re.search(r"\[ANSWER\].*?\[/ANSWER\]", pred, re.DOTALL))

        results["has_think_section"].append(has_think)
        results["has_answer_section"].append(has_answer)
        results["has_both_sections"].append(has_think and has_answer)

    metrics = {
        "think_section_percentage": np.mean(results["has_think_section"]) * 100,
        "answer_section_percentage": np.mean(results["has_answer_section"]) * 100,
        "both_sections_percentage": np.mean(results["has_both_sections"]) * 100,
    }

    return metrics


class EvalMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_preds):

        predictions = self.tokenizer.batch_decode(eval_preds, skip_special_tokens=True)

        return check_response_format(predictions)

from typing import Dict
from datasets import Dataset
import re


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def process_cot_example(
    example: Dict,
    tokenizer,
):
    question = preprocess(example["question"])
    attempt = preprocess(example["answer"])

    answer_parts = attempt.split("####")
    thinking = answer_parts[0].strip()
    answer = answer_parts[1].strip() if len(answer_parts) > 1 else ""

    assistant_text = (
        "[THINK]\n"
        + thinking
        + "\n[/THINK]\n"
        + "\n[ANSWER]\n"
        + answer
        + "\n[/ANSWER]\n"
    )

    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ],
        tokenize=False,
    )
    return dict(text=text)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    processing_function=process_cot_example,
):
    processed_dataset = dataset.map(
        lambda x: processing_function(x, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return processed_dataset

from typing import Dict
from datasets import Dataset
import re


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_cot_example(
    example: Dict,
    tokenizer,
):
    thinking_trajectory = preprocess(example["deepseek_thinking_trajectory"])
    question = preprocess(example["question"])
    answer = preprocess(example["deepseek_attempt"])

    thinking = thinking_trajectory.replace("\n\n", "\n")

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

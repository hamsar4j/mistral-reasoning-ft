from typing import Dict
from datasets import Dataset


def process_cot_example(
    example: Dict,
    tokenizer,
):
    thinking_trajectory = example["deepseek_thinking_trajectory"]
    question = example["question"]
    answer = example["deepseek_attempt"]

    thinking = thinking_trajectory.strip()
    answer = "Answer: " + answer if "Answer:" not in answer else answer

    assistant_text = "<think>\n" + thinking + "\n</think>\n" + answer.strip()

    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question.strip()},
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
    )
    return processed_dataset

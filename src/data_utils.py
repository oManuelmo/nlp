import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import MAX_LENGTH, LABEL2ID


def load_jsonl(path):
    """Load JSONL file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    """Save list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def tokenize_and_align_labels(examples, tokenizer, max_length=MAX_LENGTH, label2id=LABEL2ID):
    """
    Tokenize input tokens and align word-level BIO labels with subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []

    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special token ([CLS], [SEP], <s>, </s>, <pad>, etc.)
            if word_idx is None:
                label_ids.append(-100)
            # First subword of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            # Additional subwords of the same word
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def prepare_datasets(
    model_name,
    train_dataset=None,
    dev_dataset=None,
    test_dataset=None,
    data_dir="../data/processed",
    max_length=MAX_LENGTH,
    label2id=LABEL2ID,
):
    """
    Load tokenizer for model_name and tokenize train, dev, test datasets.
    If datasets are not passed, loads train.jsonl, dev.jsonl, test.jsonl from data_dir.
    """
    if train_dataset is None or dev_dataset is None or test_dataset is None:
        data_path = Path(data_dir)
        train_data = load_jsonl(data_path / "train.jsonl")
        dev_data = load_jsonl(data_path / "dev.jsonl")
        test_data = load_jsonl(data_path / "test.jsonl")

        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)
        test_dataset = Dataset.from_list(test_data)
    elif isinstance(train_dataset, list):
        train_dataset = Dataset.from_list(train_dataset)
        dev_dataset = Dataset.from_list(dev_dataset)
        test_dataset = Dataset.from_list(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_train = train_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, max_length=max_length, label2id=label2id
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_dev = dev_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, max_length=max_length, label2id=label2id
        ),
        batched=True,
        remove_columns=dev_dataset.column_names,
    )

    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, max_length=max_length, label2id=label2id
        ),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    return tokenizer, tokenized_train, tokenized_dev, tokenized_test

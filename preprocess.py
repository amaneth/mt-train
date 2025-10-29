# preprocess.py

import yaml
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os
import argparse


def tokenize_fn(examples, tokenizer, config):
    """Tokenize a batch of examples."""
    inputs = examples[config["datasets"]["train"]["columns"]["source"]]
    targets = examples[config["datasets"]["train"]["columns"]["target"]]
    src_langs = examples[config["datasets"]["train"]["columns"]["src_lang"]]
    tgt_langs = examples[config["datasets"]["train"]["columns"]["tgt_lang"]]

    input_ids, attention_masks, labels = [], [], []

    for src, tgt, src_lang, tgt_lang in zip(inputs, targets, src_langs, tgt_langs):
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        tokenized = tokenizer(
            src,
            text_target=tgt,
            max_length=config["tokenization"]["max_length"],
            padding=config["tokenization"]["padding"],
            truncation=config["tokenization"]["truncation"],
            return_attention_mask=True,
        )

        input_ids.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])
        labels.append(tokenized["labels"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
    }


def preprocess_data(config_path="config.yaml", output_dir="data/tokenized"):
    """Preprocess and save tokenized datasets."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # Load datasets
    train_ds = load_dataset(config["datasets"]["train"]["name"], split=config["datasets"]["train"]["split"])
    valid_ds = load_dataset(config["datasets"]["validation"]["name"], split=config["datasets"]["validation"]["split"])


    # # Test for small dataset
    # train_ds = train_ds.filter(lambda example : example["src_lang"] in ["eng_Latn", "amh_Ethi", "swh_Latn"] and example["tgt_lang"] in ["eng_Latn", "amh_Ethi", "swh_Latn"])
    # train_ds = train_ds.shuffle(seed=42).select(range(0,4000))
    # valid_ds = valid_ds.filter(lambda example : example["src_lang"] in ["eng_Latn", "amh_Ethi", "swh_Latn"] and example["tgt_lang"] in ["eng_Latn", "amh_Ethi", "swh_Latn"])
    # valid_ds = valid_ds.shuffle(seed=42).select(range(0,10))

    train_ds = train_ds.filter(lambda example: (example["src_lang"] == "eng_Latn" and example["tgt_lang"] == "amh_Ethi") or (example["src_lang"] == "amh_Ethi" and example["tgt_lang"] == "eng_Latn")) 
    valid_ds = valid_ds.filter(lambda example: (example["src_lang"] == "eng_Latn" and example["tgt_lang"] == "amh_Ethi") or (example["src_lang"] == "amh_Ethi" and example["tgt_lang"] == "eng_Latn"))

    # Tokenize
    print("Tokenizing training data...")
    tokenized_train = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer, config),
        batched=True,
        remove_columns=train_ds.column_names
    )

    print("Tokenizing validation data...")
    tokenized_valid = valid_ds.map(
        lambda x: tokenize_fn(x, tokenizer, config),
        batched=True,
        remove_columns=valid_ds.column_names
    )

    # Save processed datasets
    os.makedirs(output_dir, exist_ok=True)
    tokenized_train.save_to_disk(os.path.join(output_dir, "train"))
    tokenized_valid.save_to_disk(os.path.join(output_dir, "validation"))

    print(f"✅ Tokenized datasets saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize datasets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file.")
    args = parser.parse_args()

    preprocess_data(config_path=args.config)
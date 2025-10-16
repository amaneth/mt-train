# train.py

import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk
from huggingface_hub import HfFolder, login
import argparse
from sconf import Config

login(token=os.getenv("HF_TOKEN") or HfFolder.get_token())

def main(config_path="config.yaml", processed_data_dir="data/tokenized"):
    # Load config using sconf
    config = Config(config_path)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model.name,
        device_map=config.model.device_map,
        use_cache=config.model.use_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Load tokenized datasets
    print("Loading preprocessed datasets...")
    train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(processed_data_dir, "validation"))

    # Prepare output directory
    output_dir = config.output.dir

    # Hugging Face token (from environment or config)
    hf_token = os.getenv("HF_TOKEN")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        eval_accumulation_steps=config.training.eval_accumulation_steps,
        fp16=config.training.fp16,
        fp16_full_eval=config.training.fp16_full_eval,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        push_to_hub=config.training.push_to_hub,
        hub_private_repo=config.training.private_repo,
        hub_strategy=config.training.strategy,
        hub_token=hf_token,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train seq2seq model.")
    parser.add_argument("--tokenized_dataset", type=str, default="data/tokenized", help="Path to preprocessed dataset directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(config_path=args.config, processed_data_dir=args.tokenized_dataset)
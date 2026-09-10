import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


MODEL_NAME = "xlm-roberta-base"


def load_small_dataset():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")

    # Small subset only for testing the pipeline
    train_df = train_df.sample(
        n=min(1000, len(train_df)),
        random_state=42
    )

    val_df = val_df.sample(
        n=min(300, len(val_df)),
        random_state=42
    )

    return train_df, val_df


def main():
    print("CUDA available:", torch.cuda.is_available())

    train_df, val_df = load_small_dataset()

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]]
    )

    val_dataset = Dataset.from_pandas(
        val_df[["text", "label"]]
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    val_dataset = val_dataset.map(
        tokenize,
        batched=True
    )

    train_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    val_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    training_args = TrainingArguments(
        output_dir="models/xlm_test",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("\nStarting test training...")

    trainer.train()

    print("\nTest training completed.")

    trainer.save_model(
        "models/xlm_test/final"
    )

    tokenizer.save_pretrained(
        "models/xlm_test/final"
    )

    print(
        "Test model saved to models/xlm_test/final"
    )


if __name__ == "__main__":
    main()
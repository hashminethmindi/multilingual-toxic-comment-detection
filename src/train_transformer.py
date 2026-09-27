import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/splits/train.csv"
VALIDATION_FILE = "data/splits/val.csv"

OUTPUT_DIR = "models/xlm_final"


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class WeightedTrainer(Trainer):

    def __init__(
        self,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(
            logits.device
        )

        loss_function = nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits,
            labels
        )

        if return_outputs:
            return loss, outputs

        return loss


def calculate_class_weights(train_df):
    counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    total = len(train_df)
    num_classes = len(counts)

    weights = []

    for class_id in range(num_classes):
        class_count = counts[class_id]

        weight = (
            total /
            (num_classes * class_count)
        )

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float
    )


def clean_for_training(df, dataset_name):
    original_count = len(df)

    # Remove rows with missing text or labels
    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    # Ensure all text values are strings
    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    # Remove empty comments
    df = df[
        df["text"].str.len() > 0
    ].copy()

    # Ensure labels are integers
    df["label"] = (
        df["label"]
        .astype(int)
    )

    removed_count = (
        original_count - len(df)
    )

    print(
        f"{dataset_name}: removed "
        f"{removed_count} invalid rows"
    )

    return df


def main():

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------------
    # Load datasets
    # -----------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    # -----------------------------------
    # Clean invalid rows
    # -----------------------------------

    train_df = clean_for_training(
        train_df,
        "Train"
    )

    validation_df = clean_for_training(
        validation_df,
        "Validation"
    )

    print(
        "\nTrain rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(validation_df)
    )

    # -----------------------------------
    # Show dataset distribution
    # -----------------------------------

    print(
        "\nTraining language distribution:"
    )

    print(
        train_df[
            "language_tag"
        ].value_counts()
    )

    print(
        "\nTraining label distribution:"
    )

    print(
        train_df[
            "label"
        ].value_counts()
    )

    # -----------------------------------
    # Calculate class weights
    # -----------------------------------

    class_weights = (
        calculate_class_weights(
            train_df
        )
    )

    print(
        "\nClass weights:"
    )

    print(
        "Non-toxic (0):",
        round(
            class_weights[0].item(),
            4
        )
    )

    print(
        "Toxic (1):",
        round(
            class_weights[1].item(),
            4
        )
    )

    # -----------------------------------
    # Load tokenizer
    # -----------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # -----------------------------------
    # Load pretrained XLM-R model
    # -----------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
    )

    # -----------------------------------
    # Convert pandas DataFrames
    # to Hugging Face datasets
    # -----------------------------------

    train_dataset = (
        Dataset.from_pandas(
            train_df[
                [
                    "text",
                    "label"
                ]
            ]
        )
    )

    validation_dataset = (
        Dataset.from_pandas(
            validation_df[
                [
                    "text",
                    "label"
                ]
            ]
        )
    )

    # -----------------------------------
    # Tokenization function
    # -----------------------------------

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # -----------------------------------
    # Tokenize datasets
    # -----------------------------------

    print(
        "\nTokenizing training dataset..."
    )

    train_dataset = (
        train_dataset.map(
            tokenize,
            batched=True
        )
    )

    print(
        "Tokenizing validation dataset..."
    )

    validation_dataset = (
        validation_dataset.map(
            tokenize,
            batched=True
        )
    )

    # -----------------------------------
    # Convert dataset columns
    # to PyTorch tensors
    # -----------------------------------

    train_dataset.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ]
    )

    validation_dataset.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ]
    )

    # -----------------------------------
    # Training configuration
    # -----------------------------------

    training_args = (
        TrainingArguments(

            output_dir=OUTPUT_DIR,

            # Final full training
            num_train_epochs=2,

            learning_rate=2e-5,

            # RTX 2050 4 GB
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,

            # Effective batch size = 8
            gradient_accumulation_steps=4,

            eval_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model="f1",
            greater_is_better=True,

            # Mixed precision
            fp16=True,

            logging_steps=200,

            save_total_limit=2,

            report_to="none",

            seed=42
        )
    )

    # -----------------------------------
    # Initialize weighted trainer
    # -----------------------------------

    trainer = WeightedTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics,

        class_weights=class_weights
    )

    # -----------------------------------
    # Start training
    # -----------------------------------

    print(
        "\nStarting FULL XLM-R training..."
    )

    trainer.train()

    print(
        "\nFull training completed."
    )

    # -----------------------------------
    # Save best model
    # -----------------------------------

    best_model_path = (
        "models/xlm_final/best_model"
    )

    trainer.save_model(
        best_model_path
    )

    tokenizer.save_pretrained(
        best_model_path
    )

    print(
        "\nBest model saved to:"
    )

    print(
        best_model_path
    )


if __name__ == "__main__":
    main()
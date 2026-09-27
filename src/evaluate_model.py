import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "models/xlm_final/best_model"
TEST_FILE = "data/splits/test.csv"
RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_FILE
)

original_rows = len(test_df)

# Remove invalid rows
test_df = test_df.dropna(
    subset=["text", "label"]
).copy()

test_df["text"] = (
    test_df["text"]
    .astype(str)
    .str.strip()
)

test_df = test_df[
    test_df["text"].str.len() > 0
].copy()

test_df["label"] = (
    test_df["label"]
    .astype(int)
)

removed_rows = (
    original_rows - len(test_df)
)

print(
    "Original test rows:",
    original_rows
)

print(
    "Removed invalid rows:",
    removed_rows
)

print(
    "Final test rows:",
    len(test_df)
)

print(
    "\nTest language distribution:"
)

print(
    test_df[
        "language_tag"
    ].value_counts()
)

print(
    "\nTest label distribution:"
)

print(
    test_df[
        "label"
    ].value_counts()
)


# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------

print(
    "\nLoading trained XLM-R model..."
)

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH
    )
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

predictions = []
confidences = []

BATCH_SIZE = 16


print(
    "\nRunning predictions..."
)

texts = (
    test_df["text"]
    .tolist()
)


for start_index in range(
    0,
    len(texts),
    BATCH_SIZE
):

    batch_texts = texts[
        start_index:
        start_index + BATCH_SIZE
    ]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        batch_predictions = (
            torch.argmax(
                probabilities,
                dim=-1
            )
        )

        batch_confidences = (
            torch.max(
                probabilities,
                dim=-1
            ).values
        )

    predictions.extend(
        batch_predictions
        .cpu()
        .numpy()
        .tolist()
    )

    confidences.extend(
        batch_confidences
        .cpu()
        .numpy()
        .tolist()
    )

    processed = min(
        start_index + BATCH_SIZE,
        len(texts)
    )

    print(
        f"\rProcessed: "
        f"{processed}/{len(texts)}",
        end=""
    )


print("\nPrediction complete.")


# --------------------------------------------------
# Add predictions to dataframe
# --------------------------------------------------

test_df[
    "predicted_label"
] = predictions

test_df[
    "confidence"
] = confidences


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

true_labels = (
    test_df["label"]
    .values
)

predicted_labels = (
    test_df["predicted_label"]
    .values
)


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        zero_division=0
    )
)


print(
    "\n================================"
)

print(
    "FINAL TEST RESULTS"
)

print(
    "================================"
)

print(
    f"Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-score : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print(
    "\nClassification Report:"
)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[
        "Non-toxic",
        "Toxic"
    ],
    zero_division=0
)

print(report)


# --------------------------------------------------
# Overall metrics CSV
# --------------------------------------------------

overall_metrics = pd.DataFrame(
    [
        {
            "model": "XLM-RoBERTa",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "test_samples": len(test_df)
        }
    ]
)

overall_metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "xlm_test_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------
# Per-language metrics
# --------------------------------------------------

language_results = []

print(
    "\n================================"
)

print(
    "PER-LANGUAGE RESULTS"
)

print(
    "================================"
)


for language in sorted(
    test_df[
        "language_tag"
    ].unique()
):

    language_df = (
        test_df[
            test_df[
                "language_tag"
            ] == language
        ]
    )

    y_true = (
        language_df["label"]
        .values
    )

    y_pred = (
        language_df[
            "predicted_label"
        ].values
    )

    lang_accuracy = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    (
        lang_precision,
        lang_recall,
        lang_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    result = {
        "language": language,
        "samples": len(
            language_df
        ),
        "accuracy": lang_accuracy,
        "precision": lang_precision,
        "recall": lang_recall,
        "f1_score": lang_f1
    }

    language_results.append(
        result
    )

    print(
        f"\n{language.upper()}"
    )

    print(
        f"Samples  : "
        f"{len(language_df)}"
    )

    print(
        f"Accuracy : "
        f"{lang_accuracy:.4f} "
        f"({lang_accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: "
        f"{lang_precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{lang_recall:.4f}"
    )

    print(
        f"F1-score : "
        f"{lang_f1:.4f}"
    )


language_metrics_df = (
    pd.DataFrame(
        language_results
    )
)

language_metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "xlm_per_language_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(
    "\n================================"
)

print(
    "CONFUSION MATRIX"
)

print(
    "================================"
)

print(cm)

print(
    "\nRows    = Actual labels"
)

print(
    "Columns = Predicted labels"
)


fig = plt.figure(
    figsize=(6, 5)
)

ax = fig.add_subplot(111)

image = ax.imshow(
    cm
)

fig.colorbar(
    image
)

ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "Non-toxic",
        "Toxic"
    ]
)

ax.set_yticklabels(
    [
        "Non-toxic",
        "Toxic"
    ]
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    "XLM-R Confusion Matrix"
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "xlm_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

columns_to_save = [
    "text",
    "language_tag",
    "label",
    "predicted_label",
    "confidence"
]

test_df[
    columns_to_save
].to_csv(
    os.path.join(
        RESULTS_DIR,
        "xlm_test_predictions.csv"
    ),
    index=False
)


# --------------------------------------------------
# Error analysis files
# --------------------------------------------------

false_positives = test_df[
    (test_df["label"] == 0)
    &
    (
        test_df[
            "predicted_label"
        ] == 1
    )
]

false_negatives = test_df[
    (test_df["label"] == 1)
    &
    (
        test_df[
            "predicted_label"
        ] == 0
    )
]


false_positives.to_csv(
    os.path.join(
        RESULTS_DIR,
        "false_positives.csv"
    ),
    index=False
)

false_negatives.to_csv(
    os.path.join(
        RESULTS_DIR,
        "false_negatives.csv"
    ),
    index=False
)


print(
    "\n================================"
)

print(
    "ERROR SUMMARY"
)

print(
    "================================"
)

print(
    "False positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)


# --------------------------------------------------
# Finish
# --------------------------------------------------

print(
    "\nEvaluation completed successfully."
)

print(
    "\nResults saved inside:"
)

print(
    RESULTS_DIR
)

print(
    "\nGenerated files:"
)

print(
    "- xlm_test_metrics.csv"
)

print(
    "- xlm_per_language_metrics.csv"
)

print(
    "- xlm_confusion_matrix.png"
)

print(
    "- xlm_test_predictions.csv"
)

print(
    "- false_positives.csv"
)

print(
    "- false_negatives.csv"
)
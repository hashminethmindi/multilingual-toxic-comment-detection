import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


TRAIN_FILE = "data/splits/train.csv"
TEST_FILE = "data/splits/test.csv"

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading train and test data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


# --------------------------------------------------
# Clean data
# --------------------------------------------------

train_df = train_df.dropna(
    subset=["text", "label"]
).copy()

test_df = test_df.dropna(
    subset=["text", "label"]
).copy()

train_df["text"] = (
    train_df["text"]
    .astype(str)
    .str.strip()
)

test_df["text"] = (
    test_df["text"]
    .astype(str)
    .str.strip()
)

train_df = train_df[
    train_df["text"].str.len() > 0
].copy()

test_df = test_df[
    test_df["text"].str.len() > 0
].copy()

train_df["label"] = (
    train_df["label"]
    .astype(int)
)

test_df["label"] = (
    test_df["label"]
    .astype(int)
)


print(
    "Train rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)


# --------------------------------------------------
# Prepare features
# --------------------------------------------------

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# --------------------------------------------------
# Build baseline model
# --------------------------------------------------

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print(
    "\nTraining TF-IDF + Logistic Regression..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

print(
    "\nRunning predictions..."
)

predictions = model.predict(
    X_test
)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )
)


print(
    "\n================================"
)

print(
    "FINAL BASELINE TEST RESULTS"
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
# Save overall metrics
# --------------------------------------------------

overall_results = pd.DataFrame(
    [
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "test_samples": len(test_df)
        }
    ]
)

overall_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_baseline_metrics.csv"
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
    "PER-LANGUAGE BASELINE RESULTS"
)

print(
    "================================"
)


for language in sorted(
    test_df[
        "language_tag"
    ].unique()
):

    language_df = test_df[
        test_df[
            "language_tag"
        ] == language
    ]

    y_true = language_df["label"]

    language_predictions = model.predict(
        language_df["text"]
    )

    lang_accuracy = accuracy_score(
        y_true,
        language_predictions
    )

    (
        lang_precision,
        lang_recall,
        lang_f1,
        _
    ) = precision_recall_fscore_support(
        y_true,
        language_predictions,
        average="binary",
        zero_division=0
    )

    language_results.append(
        {
            "language": language,
            "samples": len(language_df),
            "accuracy": lang_accuracy,
            "precision": lang_precision,
            "recall": lang_recall,
            "f1_score": lang_f1
        }
    )

    print(
        f"\n{language.upper()}"
    )

    print(
        "Samples  :",
        len(language_df)
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


language_results_df = pd.DataFrame(
    language_results
)

language_results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_baseline_per_language.csv"
    ),
    index=False
)


print(
    "\nBaseline evaluation completed."
)

print(
    "Results saved to:"
)

print(
    "results/final_baseline_metrics.csv"
)

print(
    "results/final_baseline_per_language.csv"
)
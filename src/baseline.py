import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("data/processed/sample_dataset_clean.csv")

print("Dataset loaded successfully")
print("Rows:", len(df))

X = df["text"]
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)


# Create model pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Train
print("Training model...")

model.fit(X_train, y_train)

print("Training completed")


# Predict
predictions = model.predict(X_test)


# Evaluate
print("\nEvaluation Results:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Test custom comment
test_comment = ["You are stupid"]

prediction = model.predict(test_comment)[0]

if prediction == 1:
    print("Prediction: Toxic")
else:
    print("Prediction: Non-toxic")
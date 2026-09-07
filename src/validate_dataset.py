import pandas as pd


REQUIRED_COLUMNS = ["text", "label", "language"]

VALID_LANGUAGES = {
    "english",
    "sinhala",
    "tamil",
    "singlish",
    "tanglish"
}

VALID_LABELS = {0, 1}


def validate_dataset(file_path):
    df = pd.read_csv(file_path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:", missing_columns)
        return False

    # Check missing values
    print("\nMissing values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    # Check labels
    invalid_labels = set(df["label"].dropna().unique()) - VALID_LABELS

    if invalid_labels:
        print("\nERROR: Invalid labels:", invalid_labels)

    # Check languages
    invalid_languages = (
        set(df["language"].dropna().str.lower().unique())
        - VALID_LANGUAGES
    )

    if invalid_languages:
        print("\nERROR: Invalid languages:", invalid_languages)

    # Dataset distribution
    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nLanguage + label distribution:")
    print(
        df.groupby(["language", "label"])
        .size()
        .unstack(fill_value=0)
    )

    if not invalid_labels and not invalid_languages:
        print("\nDataset structure looks valid.")
        return True

    return False


if __name__ == "__main__":
    validate_dataset(
        "data/processed/sample_dataset_clean.csv"
    )
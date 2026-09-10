import os
import pandas as pd


DATASETS = [
    "english_clean.csv",
    "sinhala_clean.csv",
    "tamil_pure_clean.csv",
    "tanglish_clean.csv"
]

INPUT_FOLDER = "data/processed"
OUTPUT_FILE = "data/processed/final_multilingual_dataset.csv"


def merge_datasets():
    dataframes = []

    for filename in DATASETS:
        path = os.path.join(INPUT_FOLDER, filename)

        if not os.path.exists(path):
            print(f"Missing file: {filename}")
            continue

        df = pd.read_csv(path)

        required_columns = {
            "text",
            "label",
            "language_tag"
        }

        if not required_columns.issubset(df.columns):
            print(f"Invalid columns in {filename}")
            continue

        print(
            f"Loaded {filename}: "
            f"{len(df)} rows"
        )

        dataframes.append(df)

    if not dataframes:
        print("No valid datasets found.")
        return

    combined = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        "\nRows before duplicate removal:",
        len(combined)
    )

    combined = combined.drop_duplicates(
        subset=["text"]
    )

    print(
        "Rows after duplicate removal:",
        len(combined)
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nFinal dataset saved to:",
        OUTPUT_FILE
    )

    print("\nLanguage distribution:")
    print(
        combined["language_tag"]
        .value_counts()
    )

    print("\nLabel distribution:")
    print(
        combined["label"]
        .value_counts()
    )


if __name__ == "__main__":
    merge_datasets()
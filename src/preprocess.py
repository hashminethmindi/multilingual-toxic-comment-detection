import re
import pandas as pd


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Original rows:", len(df))

    # Remove rows with missing text
    df = df.dropna(subset=["text"])

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty comments
    df = df[df["text"].str.len() > 0]

    df.to_csv(output_path, index=False)

    print("Cleaned rows:", len(df))
    print("Saved to:", output_path)


if __name__ == "__main__":
    preprocess_dataset(
        "data/processed/sample_dataset.csv",
        "data/processed/sample_dataset_clean.csv"
    )
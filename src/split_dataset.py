import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/processed/final_multilingual_dataset.csv"

TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


def split_dataset():
    df = pd.read_csv(INPUT_FILE)

    # Combine language and label for stratification
    df["stratify_key"] = (
        df["language_tag"].astype(str)
        + "_"
        + df["label"].astype(str)
    )

    # 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["stratify_key"]
    )

    # Split remaining 30% into 15% validation and 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["stratify_key"]
    )

    # Remove helper column
    train_df = train_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])
    test_df = test_df.drop(columns=["stratify_key"])

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("Dataset split completed.")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\nTrain language distribution:")
    print(train_df["language_tag"].value_counts())

    print("\nValidation language distribution:")
    print(val_df["language_tag"].value_counts())

    print("\nTest language distribution:")
    print(test_df["language_tag"].value_counts())


if __name__ == "__main__":
    split_dataset()
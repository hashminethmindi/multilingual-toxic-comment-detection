"""Validate a model-ready CSV without changing it.

Usage: python src/validate_dataset.py data/processed/english_clean.csv

Empty CSV fields are null. Literal strings such as "NA" and "null" remain
text. Labels and language tags are checked exactly, without normalization.
"""

import argparse
from collections import Counter
import csv
import sys


REQUIRED_COLUMNS = ("text", "label", "language_tag")
VALID_LABELS = {"0", "1"}
VALID_LANGUAGES = {"english", "sinhala", "tamil", "tanglish", "singlish"}


def _print_distribution(title, counts):
    """Include null values and invalid values in the reported distributions."""
    print(f"\n{title}:")
    if counts is None:
        print("  Unavailable: required column is missing or duplicated.")
    elif not counts:
        print("  (no rows)")
    else:
        for value, count in sorted(counts.items()):
            values = value if isinstance(value, tuple) else (value,)
            display = " | ".join("<NULL>" if item == "" else repr(item)
                                 for item in values)
            print(f"  {display}: {count}")


def validate_dataset(file_path):
    """Print validation findings and return True only when all checks pass."""
    print(f"File being validated: {file_path}")
    previous_field_limit = csv.field_size_limit()
    # Accept long comments up to the platform's supported CSV field limit.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            columns = next(reader, [])
            header_counts = Counter(columns)
            missing_columns = [name for name in REQUIRED_COLUMNS
                               if name not in header_counts]
            unexpected_columns = list(dict.fromkeys(
                name for name in columns if name not in REQUIRED_COLUMNS
            ))
            duplicate_columns = {name: count for name, count in header_counts.items()
                                 if count > 1}
            # Never choose arbitrarily between columns with the same name.
            indices = {name: columns.index(name) for name in REQUIRED_COLUMNS
                       if header_counts[name] == 1}
            counts = {name: Counter() for name in indices}
            language_labels = (Counter() if {"label", "language_tag"} <= indices.keys()
                               else None)
            total_rows = 0
            malformed_rows = 0
            malformed_examples = []
            for row in reader:
                total_rows += 1
                if len(row) != len(columns):
                    malformed_rows += 1
                    if len(malformed_examples) < 5:
                        malformed_examples.append(
                            f"record {total_rows} (ending at file line {reader.line_num}): "
                            f"expected {len(columns)} fields, found {len(row)}"
                        )
                values = {name: row[index] if index < len(row) else ""
                          for name, index in indices.items()}
                for name, value in values.items():
                    counts[name][value] += 1
                if language_labels is not None:
                    language_labels[(values["language_tag"], values["label"])] += 1
    except (OSError, UnicodeError, csv.Error) as error:
        print(f"ERROR: Could not read CSV: {error}")
        print("Validation: FAIL")
        return False
    finally:
        csv.field_size_limit(previous_field_limit)

    print(f"Total number of rows: {total_rows}")
    print(f"Column names: {columns}")
    print(f"Missing required columns: {missing_columns}")
    print(f"Unexpected columns: {unexpected_columns}")
    print(f"Duplicate column names (name: count): {duplicate_columns}")
    print(f"Rows with incorrect field count: {malformed_rows}")
    for example in malformed_examples:
        print(f"  ERROR: {example}")

    failed = bool(missing_columns or unexpected_columns or duplicate_columns
                  or malformed_rows)
    unavailable = "Unavailable: required column is missing or duplicated."
    text_counts = counts.get("text")
    if text_counts is None:
        print(f"Null text count: {unavailable}")
        print(f"Whitespace-only text count: {unavailable}")
        print(f"Exact duplicate text occurrences (beyond the first): {unavailable}")
    else:
        null_text = text_counts[""]
        whitespace_text = sum(count for value, count in text_counts.items()
                              if value and not value.strip())
        duplicate_text = sum(count - 1 for value, count in text_counts.items()
                             if value and count > 1)
        duplicate_groups = sum(1 for value, count in text_counts.items()
                               if value and count > 1)
        print(f"Null text count: {null_text}")
        print(f"Whitespace-only text count: {whitespace_text}")
        print(f"Exact duplicate text occurrences (beyond the first): {duplicate_text}")
        if duplicate_text:
            print(f"WARNING: {duplicate_groups} distinct non-null texts are repeated; "
                  "duplicates are warning-only.")
        failed |= bool(null_text or whitespace_text)

    for name, allowed in (("label", VALID_LABELS), ("language_tag", VALID_LANGUAGES)):
        value_counts = counts.get(name)
        if value_counts is None:
            print(f"Null {name} count: {unavailable}")
            print(f"Invalid {name} values: {unavailable}")
            continue
        null_count = value_counts[""]
        invalid_values = {value: count for value, count in sorted(value_counts.items())
                          if value and value not in allowed}
        print(f"Null {name} count: {null_count}")
        print(f"Invalid {name} values (value: count): {invalid_values}")
        failed |= bool(null_count or invalid_values)

    _print_distribution("Label distribution", counts.get("label"))
    _print_distribution("Language distribution", counts.get("language_tag"))
    _print_distribution("Language + label distribution (language_tag | label)",
                        language_labels)
    if total_rows == 0:
        print("WARNING: The CSV contains no data rows.")
    print(f"\nValidation: {'FAIL' if failed else 'PASS'}")
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_filepath", help="Path to the CSV to validate (read-only).")
    args = parser.parse_args(argv)
    # Preserve readable diagnostics even on consoles lacking a dataset's characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    return 0 if validate_dataset(args.csv_filepath) else 1


if __name__ == "__main__":
    sys.exit(main())

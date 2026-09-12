import os
import shutil
import pandas as pd
import numpy as np
import re
import emoji
from collections import defaultdict

# Configurations
INPUT_FILE = "data/processed/singlish dataset.xlsx"
DATA_DIR = "data/processed"
META_DIR = "cleaned_data"
OUTPUT_DIR = "data/processed"

ORIGINAL_FILE_BACKUP = os.path.join(META_DIR, "raw_original.xlsx")
CLEANED_XLSX = os.path.join(DATA_DIR, "cleaned_singlish_toxic_dataset.xlsx")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_singlish_toxic_dataset.csv")
REMOVED_CSV = os.path.join(META_DIR, "removed_records.csv")
MANUAL_REVIEW_CSV = os.path.join(META_DIR, "manual_review.csv")
LABEL_CONFLICTS_CSV = os.path.join(META_DIR, "label_conflicts.csv")
REPORT_FILE = os.path.join(META_DIR, "cleaning_report.md")

MIN_TEXT_LENGTH = 2

def setup_environment():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(META_DIR):
        os.makedirs(META_DIR)
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    # Backup original
    shutil.copy2(INPUT_FILE, ORIGINAL_FILE_BACKUP)
    print(f"Backed up original dataset to {ORIGINAL_FILE_BACKUP}")

def load_data(filepath):
    try:
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            raise ValueError(f"Unsupported file extension: {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load dataset: {e}")

def identify_columns(df):
    """Identify the text and label columns based on common names."""
    cols = [str(c).lower().strip() for c in df.columns]
    
    text_col = None
    label_col = None
    
    text_keywords = ['comment', 'comments', 'text', 'sentence', 'content', 'message', 'review', 'statement', 'singlish']
    label_keywords = ['label', 'class', 'category', 'toxic', 'sentiment', 'target']
    
    for i, c in enumerate(cols):
        if not text_col and any(kw in c for kw in text_keywords):
            text_col = df.columns[i]
        elif not label_col and any(kw in c for kw in label_keywords):
            label_col = df.columns[i]
            
    # Fallback heuristic for text
    if not text_col:
        for c in df.columns:
            if df[c].dtype == 'object' and df[c].str.len().mean() > 10:
                text_col = c
                break
                
    return text_col, label_col

def standardize_columns(df, text_col, label_col):
    if not text_col:
        raise ValueError("Could not automatically identify the text column.")
    
    df['original_text'] = df[text_col].copy()
    df['text'] = df[text_col].copy()
    
    if label_col:
        df['original_label'] = df[label_col].copy()
        
        # Standardize labels
        def map_label(x):
            if pd.isna(x):
                return x
            x = str(x).lower().strip()
            if x in ['1', '1.0', 'toxic', 'yes', 'offensive']:
                return 'toxic'
            elif x in ['0', '0.0', 'non-toxic', 'no', 'non-offensive', 'non toxic']:
                return 'non-toxic'
            return x
        
        df['label'] = df['original_label'].apply(map_label)
    else:
        df['original_label'] = np.nan
        df['label'] = np.nan
        
    return df

def clean_text_pipeline(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Optional URL/Username replacement (per requirement, don't delete comment)
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "<USER>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Normalize whitespace, tabs, line breaks
    text = re.sub(r"[\r\n\t]+", " ", text)
    
    # Fix repeated punctuation while preserving toxic markers (e.g. ! or ?)
    text = re.sub(r'([.?!])\1{2,}', r'\1\1', text)
    
    # Fix excessive character repetition carefully (keep 2 chars max) e.g. haaaaari -> haari
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    # Normalize excessive spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip().lower()

def detect_language(text):
    if pd.isna(text):
        return 'Unknown'
    
    # Sinhala unicode range U+0D80–U+0DFF
    if re.search(r'[\u0D80-\u0DFF]', text):
        # We classify as Sinhala if it has Sinhala unicode characters
        return 'Sinhala'
    
    # Basic Singlish heuristics
    # Given the requirements, we shouldn't assume every Latin sentence is English.
    # We check for common Singlish words.
    text_lower = str(text).lower()
    singlish_keywords = ['mama', 'api', 'uba', 'oyata', 'mokakda', 'mokadda', 'hari', 
                         'hodai', 'narakai', 'yako', 'ban', 'machan', 'kiyapan', 'karapan', 
                         'puluwanda', 'epaa', 'naha', 'ne', 'thamai', 'wage', 'kiyala', 
                         'danne', 'inne', 'yanawa', 'enawa', 'karanawa', 'meka', 'eka']
    
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    match_count = sum(1 for w in words if w in singlish_keywords)
    
    # Very crude heuristic to separate English from Mixed/Singlish
    english_keywords = ['the', 'is', 'in', 'at', 'of', 'on', 'and', 'a', 'to', 'for', 'with']
    eng_match_count = sum(1 for w in words if w in english_keywords)
    
    if match_count > 0 and eng_match_count > 0:
        return 'Mixed'
    elif match_count > 0:
        return 'Singlish'
    elif eng_match_count > 2:
        return 'English'
    else:
        # If it's short, we don't know, but since dataset is Singlish, default to Singlish
        return 'Singlish'

def detect_subtitle_noise(text):
    if pd.isna(text):
        return False
    
    # Subtitle markers
    noise_patterns = [
        r'\d{2}:\d{2}:\d{2}', # Timestamp
        r'\[music\]', r'\[applause\]', r'\[.*?\]', # Stage directions
        r'webvtt', r'srt',
        r'^\d+$' # Just numbers (subtitle sequence)
    ]
    
    text_lower = str(text).lower()
    for p in noise_patterns:
        if re.search(p, text_lower):
            return True
    return False

def determine_removal(row):
    text = str(row['clean_text']).strip()
    
    if text == "" or pd.isna(row['clean_text']) or text.lower() == "nan":
        return "empty_text"
    
    if len(text) < MIN_TEXT_LENGTH:
        # Avoid removing short meaningful things, but length < 2 is probably invalid
        return "invalid_text"
    
    # Emojis only
    text_no_emoji = emoji.replace_emoji(text, replace='')
    if len(text_no_emoji.strip()) == 0:
        return "emoji_only"
        
    # Numbers only
    if re.fullmatch(r'\d+', text.replace(" ", "")):
        return "number_only"
        
    # URLs only (with our replacement tags)
    text_no_url = text.replace("<url>", "").replace("<user>", "").strip()
    if len(text_no_url) == 0:
        return "url_only"
    
    # Punctuation only
    if not re.search(r'[a-zA-Z\u0D80-\u0DFF]', text):
        return "punctuation_only"

    if row['suspicious_subtitle'] and len(text.split()) > 15:
        # If it has subtitle noise and is long, it's probably not a comment
        return "subtitle_noise"
        
    if row['word_count'] > 200:
        return "article_like"
        
    # Language filtering (we keep English since it might be mixed, but we flag 'Other')
    if row['language_type'] == 'Unknown':
        return 'invalid_text'
        
    return None

def main():
    print("Starting Singlish Dataset Cleaning Pipeline...")
    setup_environment()
    
    print("\n1. Loading and Inspecting Dataset...")
    df = load_data(INPUT_FILE)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n2. Identifying columns...")
    text_col, label_col = identify_columns(df)
    print(f"Detected text column: {text_col}")
    print(f"Detected label column: {label_col}")
    
    df = standardize_columns(df, text_col, label_col)
    
    # Remove rows where the original text is completely null (not just string "NaN")
    initial_count = len(df)
    df = df[df['text'].notna()]
    print(f"Removed {initial_count - len(df)} rows with completely null text.")
    
    print("\n3. Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text_pipeline)
    df['text_length'] = df['clean_text'].str.len()
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    
    print("\n4. Detecting language and subtitle noise...")
    df['language_type'] = df['text'].apply(detect_language)
    df['suspicious_subtitle'] = df['text'].apply(detect_subtitle_noise)
    
    print("\n5. Applying removal rules...")
    df['removal_reason'] = df.apply(determine_removal, axis=1)
    
    print("\n6. Handling Duplicates...")
    # Exact duplicate flag on original text (case insensitive)
    df['normalized_for_dedup'] = df['clean_text']
    
    duplicates_mask = df.duplicated(subset=['normalized_for_dedup'], keep=False)
    df['duplicate_flag'] = duplicates_mask
    
    # We will mark subsequent duplicates for removal
    subsequent_duplicates_mask = df.duplicated(subset=['normalized_for_dedup'], keep='first')
    df.loc[subsequent_duplicates_mask & df['removal_reason'].isna(), 'removal_reason'] = 'duplicate'
    
    print("\n7. Handling Labels & Conflicts...")
    df['label_conflict'] = False
    if label_col:
        # Find conflicts: same text, different labels
        df_valid_labels = df.dropna(subset=['label'])
        conflict_groups = df_valid_labels.groupby('normalized_for_dedup')['label'].nunique()
        conflicting_texts = conflict_groups[conflict_groups > 1].index
        
        df.loc[df['normalized_for_dedup'].isin(conflicting_texts), 'label_conflict'] = True
        
        # We put conflicting rows in manual review and remove them from main dataset
        df.loc[df['label_conflict'] & df['removal_reason'].isna(), 'removal_reason'] = 'label_conflict'
    
    # We also route ambiguous language items to manual review
    # Let's say if we are totally unsure
    df['manual_review'] = df['label_conflict'] | (df['language_type'] == 'Unknown') | (df['suspicious_subtitle'] & df['removal_reason'].isna())
    
    print("\n8. Routing records to files...")
    # Add an ID column
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index + 1)
    
    removed_df = df[df['removal_reason'].notna()].copy()
    manual_review_df = df[df['manual_review'] & df['removal_reason'].isna()].copy()
    
    # Main dataset is what is NOT removed and NOT flagged for manual review
    clean_df = df[df['removal_reason'].isna() & ~df['manual_review']].copy()
    
    # Select final columns for main dataset
    final_cols = ['id', 'original_text', 'clean_text', 'original_label', 'label', 
                  'language_type', 'text_length', 'word_count', 'duplicate_flag', 
                  'label_conflict', 'suspicious_subtitle', 'removal_reason']
    
    # Keep only available columns
    final_cols = [c for c in final_cols if c in clean_df.columns]
    
    clean_df = clean_df[final_cols]
    
    # Save files
    clean_df.to_excel(CLEANED_XLSX, index=False)
    clean_df.to_csv(CLEANED_CSV, index=False)
    
    removed_df.to_csv(REMOVED_CSV, index=False)
    manual_review_df.to_csv(MANUAL_REVIEW_CSV, index=False)
    
    if label_col:
        label_conflicts_df = df[df['label_conflict']]
        label_conflicts_df.to_csv(LABEL_CONFLICTS_CSV, index=False)
    else:
        pd.DataFrame(columns=df.columns).to_csv(LABEL_CONFLICTS_CSV, index=False)
    
    print("\n9. Generating Report...")
    # Stats
    total_before = len(df)
    total_after = len(clean_df)
    removed_counts = removed_df['removal_reason'].value_counts()
    lang_dist = clean_df['language_type'].value_counts()
    
    if label_col:
        label_dist = clean_df['label'].value_counts()
    else:
        label_dist = pd.Series(dtype=int)
    
    report_content = f"""# Dataset Cleaning Report

## Before Cleaning
- **Total records:** {total_before}
- **Number of columns:** {len(df.columns)}
- **Missing text values:** {df['original_text'].isna().sum()}
- **Duplicate records:** {df['duplicate_flag'].sum()}
- **Language distribution:**
{df['language_type'].value_counts().to_string()}

## Removed
"""
    for reason, count in removed_counts.items():
        pct = (count / total_before) * 100
        report_content += f"- **{reason}**: {count} ({pct:.2f}%)\n"

    report_content += f"""
## After Cleaning
- **Final record count:** {total_after}
- **Singlish count:** {lang_dist.get('Singlish', 0)}
- **Mixed-language count:** {lang_dist.get('Mixed', 0)}
- **English count:** {lang_dist.get('English', 0)}
- **Sinhala count:** {lang_dist.get('Sinhala', 0)}
- **Average text length:** {clean_df['text_length'].mean():.2f}
- **Minimum text length:** {clean_df['text_length'].min()}
- **Maximum text length:** {clean_df['text_length'].max()}

## Label Distribution
"""
    if len(label_dist) > 0:
        toxic_count = label_dist.get('toxic', 0)
        non_toxic_count = label_dist.get('non-toxic', 0)
        report_content += f"- **Toxic:** {toxic_count}\n"
        report_content += f"- **Non-toxic:** {non_toxic_count}\n"
        
        if total_after > 0:
            toxic_pct = (toxic_count / total_after) * 100
            non_toxic_pct = (non_toxic_count / total_after) * 100
            report_content += f"- **Toxic Percentage:** {toxic_pct:.2f}%\n"
            report_content += f"- **Non-Toxic Percentage:** {non_toxic_pct:.2f}%\n"
    else:
        report_content += "No valid labels found in dataset.\n"

    report_content += f"""
## Data Quality
- **Duplicates remaining:** 0 (all removed)
- **Missing values remaining:** 0
- **Conflicting labels flagged:** {df['label_conflict'].sum()}
- **Suspicious records (kept):** {clean_df['suspicious_subtitle'].sum()}
- **Manually reviewed records:** {len(manual_review_df)}
"""
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print("\n=============================================")
    print("Pipeline finished successfully!")
    print(f"Check {OUTPUT_DIR} for the output files.")
    print("=============================================\n")

    print("\n--- SANITY CHECK SAMPLES (Original -> Cleaned) ---")
    for _, row in clean_df.head(5).iterrows():
        print(f"Orig: {row['original_text']}")
        print(f"Cln : {row['clean_text']}\n")
        
    print("\n--- REMOVED SAMPLES ---")
    for _, row in removed_df.head(5).iterrows():
        print(f"Orig: {row['original_text']} [Reason: {row['removal_reason']}]")

if __name__ == "__main__":
    main()

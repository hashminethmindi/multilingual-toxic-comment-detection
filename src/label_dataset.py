import os
import pandas as pd
import re

# Configurations
INPUT_CSV = "data/processed/cleaned_singlish_toxic_dataset.csv"
OUTPUT_XLSX = "data/processed/cleaned_singlish_toxic_dataset.xlsx"
OUTPUT_CSV = "data/processed/cleaned_singlish_toxic_dataset.csv"

# Comprehensive Singlish toxicity dictionary
TOXIC_WORDS = [
    r'\bponna\w*', r'\bpaka\w*', r'\bhuth\w*', r'\bhuthto\w*', r'\bhuto\w*', 
    r'\bwesi\w*', r'\bvesi\w*', r'\bvesa\w*', r'\bkari\w*', r'\bballa\w*', 
    r'\bballi\w*', r'\bhukan\w*', r'\bpari\w*', r'\bthuk\w*', r'\bpuka\w*', 
    r'\bpadaya\w*', r'\bamandap\w*', r'\brendan\w*', r'\bpako\w*', r'\bgon\w*',
    r'\bbesika\w*', r'\bawajathaka\w*', r'\bmotha\w*', r'\bkalakanni\w*', r'\bnidikin\w*'
]

def label_text(text):
    if pd.isna(text):
        return 'non-toxic'
        
    text_lower = str(text).lower()
    
    # Check for matches
    for pattern in TOXIC_WORDS:
        if re.search(pattern, text_lower):
            return 'toxic'
            
    return 'non-toxic'

def main():
    print("Loading cleaned dataset...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Applying toxicity labels to {len(df)} records...")
    
    # Apply labeling
    df['label'] = df['clean_text'].apply(label_text)
    
    # Count results
    toxic_count = (df['label'] == 'toxic').sum()
    non_toxic_count = (df['label'] == 'non-toxic').sum()
    
    print("\n--- Labeling Results ---")
    print(f"Toxic Comments:     {toxic_count} ({(toxic_count/len(df))*100:.2f}%)")
    print(f"Non-Toxic Comments: {non_toxic_count} ({(non_toxic_count/len(df))*100:.2f}%)")
    
    # Save back to CSV
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Save back to XLSX
    print(f"Saving to {OUTPUT_XLSX}...")
    df.to_excel(OUTPUT_XLSX, index=False)
    
    print("\nLabeling complete! Showing a few toxic examples:")
    toxic_samples = df[df['label'] == 'toxic']['clean_text'].head(10).tolist()
    for s in toxic_samples:
        print(f"- {s}")

if __name__ == "__main__":
    main()

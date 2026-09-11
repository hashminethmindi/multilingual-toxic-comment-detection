import pandas as pd
import re

# File එකේ සම්පූර්ණ path එක (r අකුර අනිවාර්යයි)
filepath = r'C:\Users\ASUS\Desktop\Sandali MIT\data science\singlish dataset.xlsx'
df = pd.read_excel(filepath, header=None)
text_col = df[0].astype(str)

# Text Clean කරන Function එක
def clean_text(text):
    if pd.isna(text) or text.lower() == 'nan':
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text) 
    text = re.sub(r'http\S+|www\.\S+', '', text) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning data... Please wait.")
cleaned = text_col.apply(clean_text)

# Subtitle සලකුණු සහ හිස් ඒවා ඉවත් කිරීම
cleaned = cleaned[~cleaned.str.contains(r'\[unkown\]|\[', na=False)]
cleaned = cleaned[~cleaned.str.contains(r'^\d+$', na=False)] 
cleaned = cleaned[cleaned != ""]
cleaned = cleaned[cleaned.str.len() > 10]
cleaned = cleaned.drop_duplicates()

# Subtitle වල තියෙන ග්‍රාමීය වචන තියෙන පේළි අයින් කිරීම
subtitle_keywords = ['obata', 'kumakda', 'namuth', 'eya', 'ohu', 'ohuge', 'esey', 'ese', 'nowe', 'netha', 'sitinu', 'etha', 'yuthuya', 'peminenne']
def count_subtitle_words(text):
    words = text.split()
    return sum(1 for w in words if w in subtitle_keywords)

is_subtitle = cleaned.apply(lambda x: count_subtitle_words(x) > 0)
better_dataset = cleaned[~is_subtitle]

# Error එක හැදූ තැන (15000ක් හෝ ඉතිරි උපරිම ප්‍රමාණය ලබා ගැනීම)
sample_size = min(15000, len(better_dataset))
print(f"Total clean rows available: {len(better_dataset)}")
print(f"Sampling {sample_size} rows...")

final_dataset = better_dataset.sample(n=sample_size, random_state=42)

# අලුත් Excel File එකකට Save කිරීම
output_path = r'C:\Users\ASUS\Desktop\Sandali MIT\data science\Cleaned_Singlish_Dataset.xlsx'
final_dataset.to_excel(output_path, index=False, header=['Text'])

print(f"Success! Data saved to: {output_path}")
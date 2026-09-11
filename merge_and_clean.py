import pandas as pd
import re

# 1. Read your existing labelled dataset
df_old = pd.read_excel('Labelled_Singlish_Dataset.xlsx')

# 2. Read the new 5,000 toxic comments CSV file
# (Make sure the file name matches your exact file name)
df_new = pd.read_csv('singlish_toxic_comments_5000 (1).csv')

# 3. Standardize the column name if needed
if 'comment' in df_new.columns:
    df_new['Text'] = df_new['comment']

# Text cleaning function
def clean_text(text):
    if pd.isna(text) or str(text).lower() == 'nan':
        return ""
    text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text) # Remove timestamps
    text = re.sub(r'http\S+|www\S+', '', text)    # Remove URLs
    text = re.sub(r'@\w+', '', text)              # Remove mentions
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces
    return text

# Clean the new text column and label all of them as Toxic (1)
df_new['Text'] = df_new['Text'].apply(clean_text)
df_new['Label'] = 1

# Keep only the required columns
df_new_cleaned = df_new[['Text', 'Label']]

# 4. Merge the old dataset and the newly cleaned dataset together
combined_df = pd.concat([df_old, df_new_cleaned], ignore_index=True)

# Remove any duplicate rows based on the text
combined_df = combined_df.drop_duplicates(subset=['Text']).reset_index(drop=True)

# 5. Save the combined data into updated Excel files
combined_df.to_excel('Labelled_Singlish_Dataset.xlsx', index=False)
combined_df[combined_df['Label'] == 1].to_excel('Toxic_Only.xlsx', index=False)
combined_df[combined_df['Label'] == 0].to_excel('Non_Toxic_Only.xlsx', index=False)

print("Success! The new 5000 rows have been cleaned and merged.")
print(f"Total rows now: {len(combined_df)}")
print(f"Total Toxic rows: {(combined_df['Label'] == 1).sum()}")
print(f"Total Non-Toxic rows: {(combined_df['Label'] == 0).sum()}")
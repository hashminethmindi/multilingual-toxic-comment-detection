import pandas as pd

# 1. Read your cleaned dataset
df = pd.read_excel('Cleaned_Singlish_Dataset.xlsx')

# 2. Create a list of toxic keywords (bad words)
# (You can add as many words as you want here)
toxic_keywords = ['paka', 'huth', 'kari', 'ponna', 'vesi', 'balla', 'huka', 'puka', 'pakaya']

# 3. Function to check for keywords and label the text
def check_toxicity(text):
    text = str(text).lower() # Convert all letters to lowercase (for easier matching)
    for word in toxic_keywords:
        if word in text:
            return 1 # If a bad word is found, return Toxic (1)
    return 0 # If no bad words are found, return Non-toxic (0)

# 4. Create a new 'Label' column
df['Label'] = df['Text'].apply(check_toxicity)

# 5. Separate Toxic and Non-toxic comments
toxic_comments = df[df['Label'] == 1]
non_toxic_comments = df[df['Label'] == 0]

# 6. Save as new Excel files
df.to_excel('Labelled_Singlish_Dataset.xlsx', index=False) # The complete dataset
toxic_comments.to_excel('Toxic_Only.xlsx', index=False)    # Only Toxic comments
non_toxic_comments.to_excel('Non_Toxic_Only.xlsx', index=False) # Only Non-toxic comments

# 7. Print the results
print("Done! Files have been saved successfully.")
print(f"Total number of sentences: {len(df)}")
print(f"Number of Toxic sentences: {len(toxic_comments)}")
print(f"Number of Non-toxic sentences: {len(non_toxic_comments)}")
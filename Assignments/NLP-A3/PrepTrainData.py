from sklearn.model_selection import train_test_split
import re
import pandas as pd

# File paths (downloaded from TALPCo repo)
myn = './dataset/data_myn.txt'
eng = './dataset/data_eng.txt'

# Read the files
with open(myn, 'r', encoding='utf-8') as myfile:
    myn_lines = myfile.readlines()

with open(eng, 'r', encoding='utf-8') as engfile:
    eng_lines = engfile.readlines()

# English and Myanmar lines paired into tuples
paired_lines = list(zip(myn_lines, eng_lines))

def preprocess(text):
    # Remove special characters with regular expression
    processed_text = re.sub('[^A-Za-z0-9က-၏ဠ-ဿ၀-၉၊။ ]+', '', text)
    processed_text = re.sub('\d+', '', processed_text)
    return processed_text.strip()

processed_corpus = {(preprocess(english), preprocess(myanmar)) for myanmar, english in paired_lines}

# Build dataframe for train,val,test split
df = pd.DataFrame(processed_corpus, columns=['en', 'my'])

# Split twice for 80% train, 10% validation, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save files for upload
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
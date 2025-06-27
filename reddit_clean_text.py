import pandas as pd
import re
import spacy
from tqdm import tqdm

# ========== 1. Load spaCy model (optimized for speed) ==========
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ========== 2. Define cleaning functions ==========

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    return text

def spacy_clean_doc(doc):
    return ' '.join(
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 2
    )

# ========== 3. Setup file paths and parameters ==========

input_path = '/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_spring25_filtered.csv'
output_path_chunks = '/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_cleaned_chunks.csv'
output_path_final = '/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_cleaned_FINAL.csv'

chunk_size = 1000
max_words = 500
first_chunk = True
all_clean_chunks = []

# Optional: track progress
tqdm.pandas()

# ========== 4. Process and clean the CSV in chunks ==========

for chunk in pd.read_csv(input_path, chunksize=chunk_size):
    # --- Combine text columns safely ---
    chunk['full_text'] = (
        chunk['post_title'].fillna('') + ' ' +
        chunk['post_self_text'].fillna('') + ' ' +
        chunk['self_text'].fillna('')
    )

    # --- Truncate long posts (to reduce processing time) ---
    chunk['full_text'] = chunk['full_text'].apply(lambda x: ' '.join(x.split()[:max_words]))

    # --- Basic text cleaning (regex) ---
    chunk['basic_clean'] = chunk['full_text'].progress_apply(basic_cleaning)

    # --- Apply spaCy in batch mode (fast!) ---
    docs = nlp.pipe(chunk['basic_clean'].tolist(), batch_size=32)
    chunk['clean_text'] = [spacy_clean_doc(doc) for doc in docs]

    # --- Filter out very short posts ---
    chunk = chunk[chunk['clean_text'].str.split().str.len() > 5]

    # --- Save cleaned chunk to disk incrementally ---
    chunk[['clean_text']].to_csv(
        output_path_chunks,
        mode='a',
        header=first_chunk,
        index=False
    )
    first_chunk = False

    # --- Keep in memory for final DataFrame ---
    all_clean_chunks.append(chunk[['clean_text']])

# ========== 5. Combine all chunks into one DataFrame ==========
final_cleaned_df = pd.concat(all_clean_chunks, ignore_index=True)

# ========== 6. Save final full cleaned dataset ==========
final_cleaned_df.to_csv(output_path_final, index=False)

print(" Done! Final cleaned data saved to:")
print(f"- Partial chunks: {output_path_chunks}")
print(f"- Final combined: {output_path_final}")

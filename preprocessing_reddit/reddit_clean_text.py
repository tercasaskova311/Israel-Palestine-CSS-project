import pandas as pd
import re
import spacy
from tqdm import tqdm

# ========== 1. Load spaCy model with POS tagging (for lemmatization) ==========
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # keep tagger
nlp.add_pipe("sentencizer")  #  fixes sentence boundary error
tqdm.pandas()  # for progress bars

# ========== 2. Define cleaning functions ==========

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    return text

def spacy_clean_doc(doc):
    # Keep sentence boundaries for better structure
    return '. '.join(
        ' '.join(
            token.lemma_ for token in sent
            if token.is_alpha and not token.is_stop and len(token.text) > 2
        )
        for sent in doc.sents
    )

# ========== 3. Setup file paths and parameters ==========

input_path = '../dataReddit_spring25_filtered.csv'
output_path_chunks = '../dataReddit_cleaned_chunks.csv'
output_path_final = '../dataReddit_cleaned_FINAL.csv'

chunk_size = 1000
max_words = 500
first_chunk = True
all_clean_chunks = []

# ========== 4. Process and clean the CSV in chunks ==========

with pd.read_csv(input_path, chunksize=chunk_size) as reader:
    for chunk in reader:
        # --- Combine text columns into full_text ---
        chunk['full_text'] = (
            chunk['post_title'].fillna('') + ' ' +
            chunk['post_self_text'].fillna('') + ' ' +
            chunk['self_text'].fillna('')
        ).str.strip()

        # --- Truncate long posts (max 500 words) ---
        chunk['full_text'] = chunk['full_text'].apply(lambda x: ' '.join(x.split()[:max_words]))

        # --- Apply basic regex cleaning ---
        chunk['basic_clean'] = chunk['full_text'].progress_apply(basic_cleaning)

        # --- Run spaCy lemmatization in batch mode ---
        docs = nlp.pipe(chunk['basic_clean'].tolist(), batch_size=32)
        chunk['clean_text'] = [spacy_clean_doc(doc) for doc in docs]

        # --- Save cleaned chunk with all original columns + clean_text ---
        chunk.to_csv(
            output_path_chunks,
            mode='a',
            header=first_chunk,
            index=False
        )
        first_chunk = False

        # --- Store for final full DataFrame ---
        all_clean_chunks.append(chunk)
# if we can do this in memory, we can avoid writing to disk multiple times
# ========== 5. Combine all cleaned chunks ==========
final_cleaned_df = pd.concat(all_clean_chunks, ignore_index=True)

# ========== 6. Save full final DataFrame ==========
final_cleaned_df.to_csv(output_path_final, index=False)

print("Done! Final cleaned data saved to:")
print(f"- Partial chunks: {output_path_chunks}")
print(f"- Final combined: {output_path_final}")

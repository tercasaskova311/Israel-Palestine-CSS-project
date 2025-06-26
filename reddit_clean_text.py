import pandas as pd
import re
import spacy

reddit_filtered = pd.read_csv('/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/reddit_spring_filtered.csv')
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ========== 2. Define cleaning functions ==========

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r"[^\w\s]", '', text)  # remove punctuation
    text = re.sub(r"\d+", '', text)  # remove digits
    return text

def spacy_clean(text):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 2
    ]
    return ' '.join(tokens)

def full_preprocess(text):
    if pd.isna(text): return ''
    text = basic_cleaning(text)
    return spacy_clean(text)

# ========== 3. Combine title, post, and comment into one text column ==========

reddit_filtered['full_text'] = reddit_filtered['post_title'].fillna('') + ' ' + \
                                reddit_filtered['post_self_text'].fillna('') + ' ' + \
                                reddit_filtered['self_text'].fillna('')

# ========== 4. Apply preprocessing ==========

reddit_filtered['clean_text'] = reddit_filtered['full_text'].apply(full_preprocess)

# ========== 5. Post-cleaning filtering ==========

# Drop duplicates
reddit_spring_25 = reddit_filtered.drop_duplicates(subset='clean_text')

# Drop short texts (<5 tokens)
reddit_spring_25 = reddit_filtered[reddit_filtered['clean_text'].str.split().str.len() > 5]

# Optional: Save the cleaned data
reddit_spring_25.to_csv('/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_cleaned.csv', index=False)

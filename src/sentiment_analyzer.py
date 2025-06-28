from transformers import pipeline
import pandas as pd

# Load the cleaned Reddit data
df = pd.read_csv('../data/reddit_cleaned_FINAL.csv')
# Load sentiment/emotion analysis pipeline
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True)

# Apply on your cleaned Reddit text
df['emotion'] = df['clean_text'].apply(lambda x: classifier(x[:512])[0]['label'])  # truncate to 512 tokens

# Returns a list of all emotion scores per row
df['emotion_scores'] = df['clean_text'].apply(lambda x: classifier(x[:512])[0])


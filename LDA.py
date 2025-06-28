import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

#=== LOAD + CHOOSE COL ==========
df = pd.read_csv('/Users/terezasaskova/Downloads/Reddit_cleaned_FINAL.csv')
df = df[['post_created_time', 'clean_text']]

# Vectorize ============
vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['clean_text'])

# Fit LDA model =========
lda_model = LatentDirichletAllocation(n_components=15, max_iter=12, random_state=42)
lda_model.fit(doc_term_matrix)

# Save topic distribution ======
topic_distributions = lda_model.transform(doc_term_matrix)
df['dominant_topic'] = topic_distributions.argmax(axis=1)
df['post_created_time'] = pd.to_datetime(df['post_created_time'])
df['week'] = df['post_created_time'].dt.to_period('W')

# Save files ======
df.to_csv("LDA_with_topics.csv", index=False)
np.save("topic_distributions.npy", topic_distributions)





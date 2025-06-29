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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#=== LOAD LDA ===========
df = pd.read_csv("LDA_with_topics.csv")
topic_distributions = np.load("topic_distributions.npy")

#==== CREATE WEEKLY TOPIC COL ============
df['week'] = pd.to_datetime(df['post_created_time']).dt.to_period('W')
topic_weekly = df.groupby(['week', 'dominant_topic']).size().unstack(fill_value=0)

#==== ADD ACLED =========
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv")
acled['event_date'] = pd.to_datetime(acled['event_date'])
acled['week'] = acled['event_date'].dt.to_period('W')
weekly_violence = acled.groupby('week')['fatalities'].sum()

#=== Plot LDA topics + ACLED ===
fig, ax1 = plt.subplots(figsize=(14, 6))
topic_weekly.plot(ax=ax1)
ax1.set_ylabel('Reddit Topic Frequency')
ax1.set_xlabel('Week')

ax2 = ax1.twinx()
weekly_violence.plot(ax=ax2, color='black', linestyle='--', label='ACLED Fatalities')
ax2.set_ylabel('Conflict Fatalities')

plt.title('Weekly Topic Frequencies vs. Conflict Fatalities')
fig.legend(loc='upper right')
plt.tight_layout()
plt.show()

#===REDUSE DIM: t-SNE on LDA topic distribution ===
tsne_model = TSNE(n_components=2, random_state=42, perplexity=50)
tsne_values = tsne_model.fit_transform(topic_distributions)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_values[:, 0], tsne_values[:, 1], c=df['dominant_topic'], cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Dominant Topic')
plt.title('t-SNE Visualization of Reddit Posts by LDA Topic')
plt.xlabel('TSNE-1')
plt.ylabel('TSNE-2')
plt.tight_layout()
plt.show()



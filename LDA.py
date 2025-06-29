import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# === LOAD REDDIT ===
df = pd.read_csv('/Users/terezasaskova/Downloads/reddit_cleaned_final (1).csv')
df = df[['created_time', 'clean_text']]
df['created_time'] = pd.to_datetime(df['created_time'])

# === VECTORIZATION ===
vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['clean_text'])

# === LDA ===
lda_model = LatentDirichletAllocation(n_components=15, max_iter=12, random_state=42)
lda_model.fit(doc_term_matrix)

# === TOPIC DISTRIBUTIONS ===
topic_distributions = lda_model.transform(doc_term_matrix)
df['dominant_topic'] = topic_distributions.argmax(axis=1)

# === TOPIC LABELS ===
def get_topic_labels(lda_model, vectorizer, top_n=3):
    words = np.array(vectorizer.get_feature_names_out())
    labels = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = words[topic.argsort()[-top_n:][::-1]]
        labels.append(f"Topic {topic_idx}: " + ", ".join(top_words))
    return labels

topic_labels = get_topic_labels(lda_model, vectorizer)

# === AGGREGATE BY DAY ===
df['day'] = df['created_time'].dt.to_period('D')
daily_topics = df.groupby(['day', 'dominant_topic']).size().unstack(fill_value=0)
smoothed_topics = daily_topics.rolling(window=7, min_periods=1).mean()

# === LOAD ACLED ===
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv")
acled['event_date'] = pd.to_datetime(acled['event_date'])
acled['day'] = acled['event_date'].dt.to_period('D')
daily_fatalities = acled.groupby('day')['fatalities'].sum()

# === PLOT: TOPIC TRENDS + VIOLENCE ===
fig, ax1 = plt.subplots(figsize=(14, 6))
smoothed_topics.plot(ax=ax1, legend=False)
ax1.set_ylabel('Smoothed Topic Frequency')
ax1.set_xlabel('Day')

ax2 = ax1.twinx()
daily_fatalities.plot(ax=ax2, color='black', linestyle='--', label='ACLED Fatalities')
ax2.set_ylabel('Daily Conflict Fatalities')

plt.title('Daily Reddit Topic Trends (Smoothed) vs. Conflict Fatalities')
fig.legend(loc='upper right')
plt.tight_layout()
plt.show()
# You can add: plt.savefig("topic_trends_vs_violence.png")
plt.savefig('LDA.png')
# === SAVE RESULTS FOR LATER t-SNE ===
df.to_csv("LDA_results_no_tsne.csv", index=False)
np.save("topic_distributions.npy", topic_distributions)
pd.DataFrame({'topic_id': range(len(topic_labels)), 'label': topic_labels}).to_csv("topic_labels.csv", index=False)

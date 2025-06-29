
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
lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, random_state=42)
lda_model.fit(doc_term_matrix)

# === TOPIC DISTRIBUTIONS ===
topic_distributions = lda_model.transform(doc_term_matrix)
df['dominant_topic'] = topic_distributions.argmax(axis=1)

# === TOPIC LABELS ===
def get_topic_labels(lda_model, vectorizer, top_n=10):
    words = np.array(vectorizer.get_feature_names_out())
    labels = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = words[topic.argsort()[-top_n:][::-1]]
        labels.append(", ".join(top_words))
    return labels

topic_labels = get_topic_labels(lda_model, vectorizer, top_n=10)

# === AGGREGATE REDDIT TOPICS BY DAY ===
df['day'] = df['created_time'].dt.to_period('D')
daily_topics = df.groupby(['day', 'dominant_topic']).size().unstack(fill_value=0)
daily_topics.index = daily_topics.index.to_timestamp()
smoothed_topics = daily_topics.rolling(window=7, min_periods=1).mean()

# === IDENTIFY TOP 5 TOPICS ===
top_topics = daily_topics.sum().sort_values(ascending=False).head(5).index.tolist()
short_labels = [f"Topic {i}" for i in top_topics]

# === LOAD ACLED ===
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv")
acled['event_date'] = pd.to_datetime(acled['event_date'])
acled['day'] = acled['event_date'].dt.to_period('D')
daily_fatalities = acled.groupby('day')['fatalities'].sum()
daily_fatalities.index = daily_fatalities.index.to_timestamp()
smoothed_fatalities = daily_fatalities.rolling(window=7, min_periods=1).mean()

# === TOP 5 DEADLIEST EVENTS ===
top_events = acled.sort_values("fatalities", ascending=False).drop_duplicates("event_date").head(5)

# === PLOT ===
fig, ax1 = plt.subplots(figsize=(14, 6))
colors = plt.cm.tab10.colors

# Plot top topics
for i, topic_idx in enumerate(top_topics):
    ax1.plot(smoothed_topics.index, smoothed_topics[topic_idx],
             label=short_labels[i], color=colors[i % len(colors)])

ax1.set_ylabel("Smoothed Reddit Topic Frequency")
ax1.set_xlabel("Date")

# Plot fatalities
ax2 = ax1.twinx()
ax2.plot(smoothed_fatalities.index, smoothed_fatalities, color="black", linestyle="--", label="ACLED Fatalities")
ax2.set_ylabel("Smoothed Conflict Fatalities")

# Vertical markers for ACLED events
for _, row in top_events.iterrows():
    event_date = row['event_date']
    label = f"{row['location']} ({row['fatalities']} killed)"
    if event_date in smoothed_topics.index:
        ax1.axvline(event_date, color='gray', linestyle=':', alpha=0.6)
        ax1.annotate(label, xy=(event_date, ax1.get_ylim()[1]*0.9),
                     rotation=90, fontsize=8, va='top', ha='right', color='gray')

# Legend and layout
fig.suptitle("Reddit Topic Trends vs. Conflict Fatalities (Smoothed)", fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.98))
plt.grid(True)
plt.tight_layout()
plt.show()

# === PRINT FULL TOPIC LABELS FOR REFERENCE ===
print("\n Top 5 LDA Topics Used in Plot:")
for i, label in zip(top_topics, short_labels):
    print(f"{label}: {topic_labels[i]}")

# ========= LDA ===================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

#======== DATA =========
df = pd.read_csv('/Users/terezasaskova/Downloads/Reddit_cleaned_FINAL.csv')  # use Reddit_cleaned_FINAL.csv for full run
df = df[['post_created_time', 'clean_text']]  # keep date + cleaned text

#===== VECTOZIZED WORDS
vectorizer = CountVectorizer(
    max_df=0.9,  # drop words that appear in >90% of docs
    min_df=10,   # drop words that appear in <10 docs
    stop_words='english'
)
doc_term_matrix = vectorizer.fit_transform(df['clean_text'])

#===== LDA MODEL
lda_model = LatentDirichletAllocation(
    n_components=10,       # adjust # topics
    max_iter=10,
    random_state=42
)
lda_model.fit(doc_term_matrix)


words = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda_model.components_):
    print(f"Topic {idx+1}:")
    print(" ".join([words[i] for i in topic.argsort()[-10:]]))

topic_distributions = lda_model.transform(doc_term_matrix)
df['dominant_topic'] = topic_distributions.argmax(axis=1)

# Convert time column to datetime
df['post_created_time'] = pd.to_datetime(df['post_created_time'])
df['week'] = df['post_created_time'].dt.to_period('W')

topic_weekly = df.groupby(['week', 'dominant_topic']).size().unstack(fill_value=0)
topic_weekly.plot(kind='line', figsize=(12, 6))

#===== ADD ACLED EVENTS ========
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv")
acled['event_date'] = pd.to_datetime(acled['event_date'])
acled['week'] = acled['event_date'].dt.to_period('W')

weekly_violence = acled.groupby('week')['fatalities'].sum()


#===== VIZUALIZE ==========
fig, ax1 = plt.subplots(figsize=(14, 6))

# Topic lines
topic_weekly.plot(ax=ax1)
ax1.set_ylabel('Reddit Topic Frequency')

# Secondary axis: ACLED events
ax2 = ax1.twinx()
weekly_violence.plot(ax=ax2, color='black', linestyle='--')
ax2.set_ylabel('Conflict Fatalities')
 
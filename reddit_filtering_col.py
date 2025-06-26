import pandas as pd

reddit_spring_25 = pd.read_csv('/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_spring25.csv')
reddit_spring_25 = pd.read_csv('/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/Reddit_spring25.csv')

selected_cols = [
    'post_created_time', 'post_title', 'self_text', 'post_self_text',
    'score', 'author_name'
]
reddit_spring_25 = reddit_spring_25[selected_cols]

reddit_spring_filtered = reddit_spring_25.to_csv('reddit_spring_filtered.csv', index = False)

import pandas as pd

# Load data
reddit_data = pd.read_csv('/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/Reddit_data.csv')

# Convert to datetime
reddit_data['post_created_time'] = pd.to_datetime(reddit_data['post_created_time'])

# Define date range
start_date = pd.Timestamp("2025-03-01")
end_date = pd.Timestamp("2025-06-25")

# Filter by date
filtered_df = reddit_data[
    (reddit_data['post_created_time'] >= start_date) &
    (reddit_data['post_created_time'] <= end_date)
]

# Save to new CSV
filtered_df.to_csv('Reddit_spring25.csv', index=False)


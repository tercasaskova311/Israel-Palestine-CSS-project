import pandas as pd

# Load data
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_data.csv")

# Convert event_date to datetime
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')

# Define columns to keep
columns_to_keep = [
    'event_date',
    'event_type',
    'sub_event_type',
    'actor1',
    'civilian_targeting',
    'location',
    'source',
    'notes',
    'fatalities',
    'population_best'
]

# Filter columns
acled_filtered = acled[columns_to_keep].copy()

# Fill missing civilian_targeting values with 'No'
acled_filtered['civilian_targeting'] = acled_filtered['civilian_targeting'].fillna('No')
# Replace 'Civilian targeting' with 'Yes'
acled_filtered['civilian_targeting'] = acled_filtered['civilian_targeting'].replace('Civilian targeting', 'Yes')



# Optional: add simplified event category
acled_filtered['event_category'] = acled_filtered['event_type'].map({
    "Battles": "Combat",
    "Explosions/Remote violence": "Combat",
    "Violence against civilians": "Civilian harm"
})

# Create daily summary
daily_events = (
    acled_filtered.groupby('event_date')
         .agg(event_count=('event_type', 'count'),
              fatalities=('fatalities', 'sum'))
         .reset_index()
)

# Show structure
acled_filtered.info()
# Export filtered data
acled_filtered.to_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv", index=False)
print("✅ Filtered ACLED data saved successfully.")

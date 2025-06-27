import pandas as pd

#Load data
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_data.csv")

#Convert event_date to datetime
acled['event_date'] = pd.to_datetime(acled['event_date'])


# Optional grouping: collapse to broad categories
acled['event_category'] = acled['event_type'].map({
    "Battles": "Combat",
    "Explosions/Remote violence": "Combat",
    "Violence against civilians": "Civilian harm"
})

#Create daily summary: number of events and fatalities
daily_events = (
    acled.groupby('event_date')
         .agg(event_count=('event_id_cnty', 'count'),
              fatalities=('fatalities', 'sum'))
         .reset_index()
)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import correlate
import numpy as np

# === Load ACLED Data ===
acled = pd.read_csv("data/ACLED_filtered.csv", parse_dates=["event_date"])
acled_daily = (
    acled.groupby("event_date")
         .agg(events=("event_type", "count"), fatalities=("fatalities", "sum"))
         .reset_index()
)

# === Load Reddit Data ===
reddit = pd.read_csv(
    "/Users/terezasaskova/Downloads/Reddit_cleaned_FINAL.csv",
    parse_dates=["post_created_time"],
    on_bad_lines="skip",
    engine="python"
)
reddit["date"] = reddit["post_created_time"].dt.normalize()
reddit_daily = reddit.groupby("date").size().reset_index(name="posts")
reddit_daily = reddit_daily.rename(columns={"date": "event_date"})

# === Merge ===
combined = pd.merge(acled_daily, reddit_daily, on="event_date", how="outer").fillna(0).sort_values("event_date")
combined.set_index("event_date", inplace=True)

# === 1. Normalized Comparison ===
scaler = MinMaxScaler()
normalized = scaler.fit_transform(combined[["events", "posts"]])
combined_normalized = pd.DataFrame(normalized, columns=["events", "posts"], index=combined.index)

plt.figure(figsize=(12, 6))
plt.plot(combined_normalized.index, combined_normalized["events"], label="ACLED Events (Normalized)")
plt.plot(combined_normalized.index, combined_normalized["posts"], label="Reddit Posts (Normalized)")
plt.title("Normalized Volume Comparison: ACLED vs. Reddit")
plt.xlabel("Date")
plt.ylabel("Normalized Daily Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. Dual Y-Axis Comparison ===
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(combined.index, combined["events"], color="blue", label="ACLED Events")
ax1.set_ylabel("ACLED Events", color="blue")

ax2 = ax1.twinx()
ax2.plot(combined.index, combined["posts"], color="orange", label="Reddit Posts")
ax2.set_ylabel("Reddit Posts", color="orange")

plt.title("Dual Axis Volume Comparison: ACLED vs. Reddit")
fig.tight_layout()
plt.grid(True)
plt.show()

# === 3. Cross-Correlation ===
acled_norm = (combined["events"] - combined["events"].mean()) / combined["events"].std()
reddit_norm = (combined["posts"] - combined["posts"].mean()) / combined["posts"].std()

correlation = correlate(reddit_norm, acled_norm, mode='full')
lags = np.arange(-len(acled_norm) + 1, len(acled_norm))

plt.figure(figsize=(12, 6))
plt.plot(lags, correlation)
plt.title("Cross-Correlation: Reddit vs. ACLED")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.axvline(0, color='black', linestyle='--', label="Zero Lag")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

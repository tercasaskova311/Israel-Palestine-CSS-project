import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import correlate
import numpy as np
import matplotlib.dates as mdates

# === Setup ===
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120  # Higher quality plots

# === Load & Preprocess ===
acled = pd.read_csv("/Users/terezasaskova/Desktop/Israel-Palestine-CSS-project/data/ACLED_filtered.csv", parse_dates=["event_date"])
reddit = pd.read_csv("/Users/terezasaskova/Downloads/reddit_cleaned_final (1).csv", parse_dates=["created_time"], on_bad_lines="skip", engine="python")

# Align Reddit date range to ACLED
max_acled_date = acled["event_date"].max()
reddit = reddit[reddit["created_time"] <= max_acled_date]

# Aggregate ACLED
acled_daily = (
    acled.groupby("event_date")
         .agg(events=("event_type", "count"), fatalities=("fatalities", "sum"))
         .reset_index()
)

# Reddit aggregation
reddit["date"] = reddit["created_time"].dt.normalize()
reddit_daily = reddit.groupby("date").size().reset_index(name="posts")
reddit_daily.rename(columns={"date": "event_date"}, inplace=True)

# Merge datasets
combined = pd.merge(acled_daily, reddit_daily, on="event_date", how="outer").fillna(0).sort_values("event_date")
combined.set_index("event_date", inplace=True)

# Rolling smoothing
combined_smoothed = combined.rolling(window=7, center=True).mean().dropna()

# Normalize
scaler = MinMaxScaler()
normalized = scaler.fit_transform(combined_smoothed[["events", "posts"]])
combined_normalized = pd.DataFrame(normalized, columns=["ACLED Events", "Reddit Posts"], index=combined_smoothed.index)

# Top 5 tragic events
tragic_events = (
    acled[(acled["event_type"].str.contains("violence", case=False)) & (acled["fatalities"] > 0)]
    .sort_values(by="fatalities", ascending=False)
    .drop_duplicates(subset=["event_date"])
    .head(5)
)

tragic_dict = dict(zip(
    tragic_events["event_date"],
    tragic_events["location"] + " (" + tragic_events["fatalities"].astype(str) + " killed)"
))

# === Plot 1: Normalized Volume ===
plt.figure(figsize=(14, 6))
plt.plot(combined_normalized.index, combined_normalized["ACLED Events"], label="ACLED Events")
plt.plot(combined_normalized.index, combined_normalized["Reddit Posts"], label="Reddit Posts")

# Annotate tragedies
for date, label in tragic_dict.items():
    if date in combined_normalized.index:
        plt.axvline(date, color='black', linestyle=':', alpha=0.6)
        plt.text(date, 0.95, label, rotation=90, verticalalignment='top', fontsize=8, ha='right', color='black')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.title("Smoothed & Normalized: ACLED vs. Reddit")
plt.xlabel("Date")
plt.ylabel("Normalized Volume")
plt.legend()
plt.tight_layout()
plt.show()

# === Plot 2: Dual Axis ===
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(combined_smoothed.index, combined_smoothed["events"], color="blue", label="ACLED Events")
ax1.set_ylabel("ACLED Events", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(combined_smoothed.index, combined_smoothed["posts"], color="orange", label="Reddit Posts")
ax2.set_ylabel("Reddit Posts", color="orange")
ax2.tick_params(axis='y', labelcolor='orange')

# Annotate vertical lines
for d, label in tragic_dict.items():
    if d in combined_smoothed.index:
        ax1.axvline(d, color='black', linestyle=':', alpha=0.6)
        ax1.annotate(label, xy=(d, ax1.get_ylim()[1]*0.8), xycoords='data',
                     rotation=90, fontsize=8, va='top', ha='right', color='black')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate()

plt.title("Smoothed Volume Comparison: ACLED (left) vs. Reddit (right)")
plt.grid(True)
fig.tight_layout()
plt.show()

# === Plot 3: Cross-Correlation ===
acled_norm = (combined_smoothed["events"] - combined_smoothed["events"].mean()) / combined_smoothed["events"].std()
reddit_norm = (combined_smoothed["posts"] - combined_smoothed["posts"].mean()) / combined_smoothed["posts"].std()

# Drop NaNs before correlate
acled_norm = acled_norm.dropna()
reddit_norm = reddit_norm.dropna()

correlation = correlate(reddit_norm, acled_norm, mode='full')
lags = np.arange(-len(acled_norm) + 1, len(acled_norm))
max_lag = lags[np.argmax(correlation)]
max_corr_value = np.max(correlation)

plt.figure(figsize=(12, 6))
plt.plot(lags, correlation, label="Cross-Correlation")
plt.axvline(0, color='black', linestyle='--', label="Zero Lag")
plt.axvline(max_lag, color='green', linestyle=':', label=f"Peak Corr @ {max_lag} days")
plt.scatter([max_lag], [max_corr_value], color='green')
plt.text(max_lag, max_corr_value, f"{round(max_corr_value, 2)}", fontsize=9, ha='right')

plt.title("Cross-Correlation: Reddit vs. ACLED (7-day Smoothed)")
plt.xlabel("Lag (days) — Positive = Reddit lags behind")
plt.ylabel("Correlation (z-scored)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr

# === 1. Descriptive Statistics ===
acled_mean = combined_smoothed["events"].mean()
acled_max = combined_smoothed["events"].max()
reddit_mean = combined_smoothed["posts"].mean()
reddit_max = combined_smoothed["posts"].max()

# === 2. Pearson Correlation (Smoothed) ===
r_pearson, p_pearson = pearsonr(combined_smoothed["events"], combined_smoothed["posts"])

# === 3. Cross-Correlation (already computed)
# max_lag and max_corr_value already exist from your code

# Also get correlation at lag = 0
zero_lag_corr = correlation[lags == 0][0]

# === 4. Salience around top 5 events ===
reddit_daily.set_index("event_date", inplace=True)

event_response = []
for date in tragic_events["event_date"]:
    start = date - pd.Timedelta(days=1)
    end = date + pd.Timedelta(days=2)
    local_max = reddit_daily.loc[start:end, "posts"].max() if not reddit_daily.loc[start:end, "posts"].empty else 0
    event_response.append((date.strftime('%Y-%m-%d'), local_max))

# === 5. Final Report-Ready Output ===

print("\n📄 Report Paragraph:\n")

print(f"""
To explore the relationship between conflict events and online discourse, we compared smoothed daily frequencies of ACLED events and Reddit posts. The average daily number of ACLED-reported conflict events was {acled_mean:.2f} (max = {acled_max:.0f}), while Reddit post volume averaged {reddit_mean:.2f} (max = {reddit_max:.0f}).

Pearson correlation between the two smoothed series was r = {r_pearson:.2f} (p = {p_pearson:.4f}), suggesting a moderate relationship. A cross-correlation analysis revealed a peak correlation of r = {max_corr_value:.2f} at a lag of {max_lag} days, indicating that Reddit discussion tends to {('follow' if max_lag > 0 else 'precede' if max_lag < 0 else 'align with')} conflict activity by about {abs(max_lag)} day(s). Correlation at lag 0 was r = {zero_lag_corr:.2f}.

Reddit activity also appeared to respond to major violent events. For example:""")

for date, peak_posts in event_response:
    print(f" - On or near {date}, Reddit reached {peak_posts} posts per day.")

print("\nWhile these findings suggest some alignment between real-world events and online discourse, causality should not be assumed without further modeling or qualitative analysis.")

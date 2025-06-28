# **Volume Comparison:** exploratory data analysis and temporal comparison between:

* **Conflict event data** from [ACLED](https://acleddata.com/)
* **Reddit discourse** from a Kaggle dataset on Israel–Palestine-related posts

---

We examine the **relationship between real-world conflict activity** and **online public discourse**, using:

* Time-series smoothing (rolling averages)
* Normalization for scale comparability
* Dual-axis plots
* **Cross-correlation** analysis to detect **temporal lags**
* Annotation of **major conflict events** and **external communication spikes** (mailer events)

---

## Files and Scripts

| File                       | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| `frequeny_comparisson.py`  | Main script for plotting time series and cross-correlation |
| `ACLED_filtered.csv`       | Cleaned and filtered conflict event dataset                |
| `Reddit_cleaned_FINAL.csv` | Cleaned Reddit posts dataset                               |
| `plots/`                   | Contains exported visualizations                           |
| `README.md`                | Project overview and insights                              |

---

## Methodology Overview

### Data Sources

* ACLED: Real-world conflict events in Gaza & West Bank
* Reddit: Public discussion related to the conflict (March–June 2025)

### Steps Taken

1. **Preprocessing**:

   * Aggregated Reddit posts and ACLED events daily
   * Applied a 7-day centered rolling average for noise reduction
   * Normalized time series to 0–1 range for visual comparison

2. **Major Event Annotation**:

   * Identified top 5 most fatal conflict events (civilian-targeted)
   * Added vertical lines and labels on plots
   * Added “mailer events” (e.g., external statements, trigger points)

3. **Cross-Correlation Analysis**:

   * Standardized both time series (mean=0, std=1)
   * Used `scipy.signal.correlate` to compute lag-based similarity
   * Annotated the peak lag (max correlation) to interpret timing

---

## Key Visualizations

* **Normalized Comparison**: Smoothed, scale-adjusted trends of ACLED vs. Reddit
* **Dual Y-Axis View**: Absolute volumes with independent y-scales
* **Cross-Correlation**:

  * Helps reveal **how Reddit discussions lag (or lead) real-world conflict**
  * For example, a peak at `+7` means Reddit tends to react **7 days after** events

---

## Potential Takeaways for LDA Topic Modeling

This cross-correlation analysis helps us define **when Reddit was most reactive** to conflict events.

These insights will guide the next phase:

###  LDA Topic Modeling Plan

* Focus on **Reddit posts around tragic events** (±3–5 days)
* Apply **LDA (Latent Dirichlet Allocation)** to extract themes
* Compare **topics across high-conflict vs. low-conflict periods**
* Plot **topic trends over time** to map them to the conflict timeline

---

## Dependencies

```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn gensim
```
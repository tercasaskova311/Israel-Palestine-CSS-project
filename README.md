# Israel–Palestine Conflict & Reddit Discourse (CSS Project)

This project analyzes how public discussion on Reddit responds to real-world conflict events in Gaza and the West Bank. We combine data from **Reddit** (online discourse) and **ACLED** (verified conflict events) to examine temporal relationships using Python-based data science methods.

---

## Project Structure
─ data/ # Cleaned datasets (Reddit, ACLED)

─ notebooks/ # Python scripts for analysis

─ plots/ Final visualizations

─ notes/ Project description for given parts

─ README.md Project overview

---

##  Research Question

> **To what extent does Reddit discourse — measured by post volume, dominant topics, and sentiment — align with verified real-world conflict events in Palestine?**

---

## Method Summary

### Data Sources

- **Reddit Posts** from [Kaggle](https://www.kaggle.com/datasets/asaniczka/reddit-on-israel-palestine-daily-update)  
- **Conflict Events** from [ACLED](https://acleddata.com/)

###  Preprocessing

- Reddit text cleaned with regex and lemmatized (spaCy)
- ACLED data filtered by date and location
- Created `event_category` column to classify events as:
  - `"Combat"` → Battles and Explosions/Remote violence
  - `"Civilian Harm"` → Violence against civilians

### Analysis Techniques

- Smoothed daily volumes using 7-day centered rolling average
- Cross-correlation to assess time-lag between online and offline activity
- Topic modeling with Latent Dirichlet Allocation (LDA)
- sentiment analysis

---
## Key Findings

- **Volume Lag**:  
  Reddit comment spikes follow real-world conflict spikes with a ~15-day delay  
  *(Highest cross-correlation: r = 0.56 at lag +15)*

- **Thematic Shifts**:  
  Reddit topics intensified around:
  - **Military conflict** (e.g., May 18, March 18)
  - **Humanitarian crises** (e.g., June 17 airstrike on aid convoy)
  - **Geopolitical blame** (e.g., Topic 3: Iran, Trump, Netanyahu)

- **Selective Engagement**:  
  Reddit attention is not constant—posts surge after major fatalities but not all conflict events.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tercasaskova311/Israel-Palestine-CSS-project.git


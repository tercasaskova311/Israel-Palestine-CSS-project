# Israel–Palestine CSS Project

We use two datasets:

1. **Reddit Posts** from Kaggle – covering public discourse around the Israel–Palestine conflict.
2. **Conflict Events** from ACLED – providing structured, real-world records of violence in Gaza.

---

## Datasets Used

### 1. Reddit Dataset (Kaggle)

- **Source**: [Reddit on Israel–Palestine – Kaggle](https://www.kaggle.com/datasets/asaniczka/reddit-on-israel-palestine-daily-update)
- **Filtered Date Range**: `2025-03-01` to `2025-06-25`
- **Filtered Columns**: post_created_time, post_title, self_text, post_self_text, score, author_name, full_text,  basic_clean, clean_text

- for using the csv file, read instruction below or use 'reddit_sample.csv' for testing 

### 2. ACLED Dataset (Real time events)

- **Source**: [ACLED Gaza Monitor] ()
- **Filtered Date Range**: `2025-03-01` to `2025-06-25`
- **Retained Columns (filtered)**:
- event_date','event_type','sub_event_type','actor1','civilian_targeting','location','source','notes',fatalities','population_best'.

Dropped Columns:
- iso, event_id_cnty, data_id, etc.	Internal ACLED metadata, not needed for analysis
- timestamp	Too granular (datetime), event_date is sufficient
-  source_scale	Useful for journalistic traceability but not for frame/sentiment comparison
- actor 2 - the other side of conflict + assoc_actor
- inter1, interacton
- year, region, country, location, lat + longtitude, geo_precision = we only analysi conlict in Gaza and West Bank, Palestine
- disorder_type


- Creating event category column:  
    - Battles & Explosions/Remote violence  = Combat
    - Violence against civilians = Civilian harm


# Cleaned Reddit Data

### Sample for Testing
- `reddit_sample.csv`: A small sample for fast loading and testing.

### Full Cleaned Dataset
- File: `Reddit_cleaned_FINAL.csv`
- Size: ~2.1 GB
- Download here: [Google Drive Link](https://drive.google.com/file/d/1PIOZe5zjtaZ9X1lrVZLzEw4GIIB3PT_A/view?usp=sharing)

You can also download it using Python:

```python
!pip install gdown
!gdown 'https://drive.google.com/file/d/1PIOZe5zjtaZ9X1lrVZLzEw4GIIB3PT_A/view?usp=sharing'


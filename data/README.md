# Data Directory

## Download the CICIoT2023 Dataset

1. Go to: https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Download all CSV files
3. Place them inside `data/raw/` (any folder structure is fine — the loader scans recursively)

## Expected Structure After Download

```
data/
├── raw/
│   ├── CICIoT2023_part1.csv
│   ├── CICIoT2023_part2.csv
│   └── ... (all CSV files)
└── processed/            ← auto-created by the pipeline
    ├── merged.parquet
    ├── dataset_labelled.parquet
    ├── splits.pkl
    └── split_indices.pkl
```

## Dataset Facts

- ~45 million network flow records
- 46+ numeric features per record (packet counts, byte counts, flow durations, flags, etc.)
- 34 classes: 1 benign + 33 attack types
- Heavy class imbalance: benign ~1M rows, some attacks only ~12K rows

## Label Column

The raw CSV files contain a column named `Label` (capital L).
The pipeline normalises it to lowercase `label` automatically.
All label strings are converted to UPPERCASE for consistency.

## Why data/ is gitignored

The raw dataset is ~40GB+. It cannot be committed to GitHub.
Only `data/processed/split_indices.pkl` should optionally be committed
to ensure reproducible splits across team members.

## Demo Mode (No Download Needed)

Set `USE_DEMO = True` in any notebook, or run:
```bash
python run_pipeline.py
```
This generates synthetic data that mirrors the real dataset structure.

"""
loader.py  (v2 — fixed for real CICIoT2023 data)
=================================================
Handles 63 Merged CSV files correctly.

KEY INSIGHT from data inspection of Merged52.csv:
  - Each file already contains all 34 classes (verified: 34/34 present)
  - Class counts differ per file (e.g. DDOS dominates, XSS has only 2-3 rows)
  - The label column is named 'Label' (capital L) in the real files
  - Duplicate rows: ~5% per file (handled in preprocessing.py)

IMBALANCE STRATEGY across 63 files:
  - Load ALL files first (concat everything)
  - Deduplicate globally (not per-file) — duplicates may span file boundaries
  - Sample AFTER global dedup using stratified per-class sampling
  - This guarantees rare classes (XSS ~189 total, BruteForce ~1,134 total)
    are never accidentally excluded by per-file sampling
  - The split is then done on the GLOBAL pool with stratification on
    the 34-class label — this is the ONLY correct way to handle this
"""

import logging
import os
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Real dataset label column name
RAW_LABEL_COL = "Label"


def build_merged_parquet(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/merged.parquet",
) -> None:
    """
    Merge all 63 Merged*.csv files into a single Parquet file.
    Run once. Skip if Parquet already exists.

    What this does:
      - Finds all CSV files in raw_dir (recursively)
      - Strips whitespace from column names
      - Normalises label column to lowercase 'label'
      - Normalises label strings to UPPERCASE
      - Concatenates all files
      - Saves as Parquet (10x faster to reload than CSV)
    """
    if os.path.exists(output_path):
        log.info(f"Parquet already exists at {output_path}. Skipping merge.")
        return

    csv_files = sorted(glob.glob(os.path.join(raw_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Place your 63 Merged*.csv files inside data/raw/ then re-run."
        )

    log.info(f"Found {len(csv_files)} CSV files. Merging...")
    frames = []

    for i, path in enumerate(csv_files, 1):
        log.info(f"  [{i:02d}/{len(csv_files)}] {os.path.basename(path)}")
        df = pd.read_csv(path, low_memory=False)

        # Normalise column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]

        # Find the label column (handles 'Label', 'label', 'LABEL')
        label_match = [c for c in df.columns if c.lower() == "label"]
        if not label_match:
            log.warning(f"  No label column found in {path}. Skipping.")
            continue
        df = df.rename(columns={label_match[0]: "label"})

        # Normalise label strings to UPPERCASE (some files may have mixed case)
        df["label"] = df["label"].astype(str).str.strip().str.upper()

        frames.append(df)

    if not frames:
        raise ValueError("No valid CSV files could be loaded.")

    merged = pd.concat(frames, ignore_index=True)
    log.info(f"Total merged shape: {merged.shape}")
    log.info(f"Classes: {merged['label'].nunique()} unique labels")
    log.info(f"Label distribution:\n{merged['label'].value_counts().to_string()}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info(f"Saved merged Parquet -> {output_path}")


def load_dataset(
    parquet_path: str = "data/processed/merged.parquet",
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the merged dataset from Parquet.

    Parameters
    ----------
    parquet_path  : path to merged Parquet file (created by build_merged_parquet)
    sample_frac   : fraction of rows to keep PER CLASS after deduplication.
                    None = keep all rows.
                    Recommended: 0.25 for local/Colab Pro, 0.10 for Colab free tier.
    random_state  : reproducibility seed

    Returns
    -------
    pd.DataFrame  with 'label' column containing UPPERCASE strings

    NOTE: Deduplication (deduplicate()) should be called AFTER this function
    and BEFORE create_splits(). See notebook 02_preprocessing.py.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Merged Parquet not found at {parquet_path}.\n"
            f"Run build_merged_parquet() first."
        )

    log.info(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded shape: {df.shape}")

    # Ensure label column exists
    if "label" not in df.columns:
        matches = [c for c in df.columns if c.lower() == "label"]
        if matches:
            df = df.rename(columns={matches[0]: "label"})
        else:
            raise KeyError(f"No label column found. Columns: {list(df.columns)[:10]}")

    df["label"] = df["label"].astype(str).str.strip().str.upper()

    if sample_frac is not None:
        df = sample_stratified(df, frac=sample_frac, random_state=random_state)

    log.info(f"Final loaded shape: {df.shape}")
    return df


def sample_stratified(
    df: pd.DataFrame,
    frac: float = 0.10,
    label_col: str = "label",
    random_state: int = 42,
    min_per_class: int = 200,
) -> pd.DataFrame:
    """
    Sample `frac` of rows per class.

    Critical for rare classes: guarantees at least `min_per_class` rows
    are kept for every class that has enough rows, regardless of frac.

    For very rare classes (< min_per_class rows total), ALL rows are kept.

    This means XSS (~189 rows total) always gets all its rows kept.
    DICTIONARYBRUTEFORCE (~1,134 rows) keeps at least 200.

    Parameters
    ----------
    df             : full deduplicated dataframe
    frac           : fraction to sample per class
    label_col      : label column name
    random_state   : seed
    min_per_class  : minimum rows to keep per class regardless of frac

    Returns
    -------
    Stratified-sampled DataFrame, shuffled.
    """
    rng = np.random.RandomState(random_state)
    frames = []

    class_counts = df[label_col].value_counts()
    log.info(f"Sampling {frac*100:.0f}% per class (min {min_per_class} per class):")

    for cls, total in class_counts.items():
        n = max(min_per_class, int(total * frac))
        n = min(n, total)   # never exceed what we have
        sampled = df[df[label_col] == cls].sample(n=n, random_state=rng.randint(0, 99999))
        frames.append(sampled)
        if total < min_per_class:
            log.info(f"  {cls:<40} {total:>7,} total -> KEPT ALL {n:>7,} (rare class)")
        else:
            log.info(f"  {cls:<40} {total:>7,} total -> sampled {n:>7,}")

    result = pd.concat(frames, ignore_index=True)
    result = result.reset_index(drop=True)
    log.info(f"Sampled total: {len(result):,} rows")
    return result


def generate_demo_data(n_samples: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that mirrors the REAL CICIoT2023 column structure.
    Uses the actual column names from Merged52.csv.
    Set USE_DEMO = True in notebooks to use this without downloading the dataset.
    """
    from src.data.label_mapping import ALL_LABELS

    rng = np.random.RandomState(random_state)

    # REAL column names from CICIoT2023 (39 features + label)
    numeric_features = [
        "Header_Length", "Time_To_Live", "Rate",
        "fin_flag_number", "syn_flag_number", "rst_flag_number",
        "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
        "ack_count", "syn_count", "fin_count", "rst_count",
        "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
        "Tot sum", "Min", "Max", "AVG", "Std",
        "Tot size", "IAT", "Number",
        # NOTE: Variance intentionally excluded (= Std^2, dropped in preprocessing)
    ]
    # Protocol Type as categorical
    protocol_values = [0, 1, 6, 17, 47]

    class_counts = {
        "BENIGN": 3000,
        "DDOS-TCP_FLOOD": 2200, "DDOS-SYN_FLOOD": 2000, "DDOS-UDP_FLOOD": 1900,
        "DDOS-ICMP_FLOOD": 1700, "DDOS-HTTP_FLOOD": 500, "DDOS-PSHACK_FLOOD": 600,
        "DDOS-ACK_FRAGMENTATION": 400, "DDOS-SLOWLORIS": 100,
        "DDOS-RSTFINFLOOD": 650, "DDOS-UDP_FRAGMENTATION": 420,
        "DDOS-SYNONYMOUSIP_FLOOD": 560, "DDOS-ICMP_FRAGMENTATION": 280,
        "DOS-TCP_FLOOD": 900, "DOS-SYN_FLOOD": 750, "DOS-UDP_FLOOD": 700,
        "DOS-HTTP_FLOOD": 120,
        "MIRAI-GREETH_FLOOD": 500, "MIRAI-GREIP_FLOOD": 430, "MIRAI-UDPPLAIN": 380,
        "RECON-HOSTDISCOVERY": 220, "RECON-OSSCAN": 130,
        "RECON-PINGSWEEP": 40, "RECON-PORTSCAN": 110, "VULNERABILITYSCAN": 160,
        "DNS_SPOOFING": 200, "MITM-ARPSPOOFING": 170,
        "XSS": 50, "SQLINJECTION": 45, "COMMANDINJECTION": 40,
        "BROWSERHIJACKING": 35, "UPLOADING_ATTACK": 30,
        "BACKDOOR_MALWARE": 25, "DICTIONARYBRUTEFORCE": 60,
    }

    frames = []
    for label, count in class_counts.items():
        proto_choices = rng.choice(protocol_values, size=count)
        X = rng.exponential(scale=20, size=(count, len(numeric_features)))

        if "DDOS" in label or "DOS" in label:
            X[:, 2] *= 50       # high Rate
            X[:, 0] *= 0.5      # shorter Header_Length
        elif "MIRAI" in label:
            X[:, 2] *= 20
        elif "RECON" in label or "SCAN" in label:
            X[:, 2] *= 5
        elif label in ("XSS", "SQLINJECTION", "COMMANDINJECTION",
                       "BROWSERHIJACKING", "UPLOADING_ATTACK", "BACKDOOR_MALWARE"):
            X[:, 2] *= 2

        frame = pd.DataFrame(X, columns=numeric_features)
        frame["Protocol Type"] = proto_choices
        frame["label"] = label
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    log.info(f"Generated synthetic demo dataset: {df.shape}")
    return df


def sample_balanced(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
    floor: int = 2000,
    ceiling: int = 80_000,
) -> pd.DataFrame:
    """
    Targeted oversampling + capped undersampling.
    Applied ONLY to the training split after create_splits().
    Val and test always keep real rows only.

    - Classes with fewer than floor rows: oversample WITH replacement to floor
    - Classes with more than ceiling rows: undersample WITHOUT replacement to ceiling
    - Classes between floor and ceiling: kept as-is

    Parameters
    ----------
    df           : training split dataframe (NOT the full dataset)
    label_col    : label column name
    random_state : seed
    floor        : minimum rows per class after balancing
    ceiling      : maximum rows per class after balancing
    """
    rng    = np.random.RandomState(random_state)
    frames = []
    class_counts = df[label_col].value_counts()

    log.info(f"Balanced sampling (TRAIN only): floor={floor:,}  ceiling={ceiling:,}")
    log.info(f"{'Class':<42} {'Real':>8} {'Sampled':>8}  Action")
    log.info("-" * 72)

    n_over = n_under = n_keep = 0

    for cls, real_count in class_counts.items():
        class_df = df[df[label_col] == cls]
        seed     = rng.randint(0, 99_999)

        if real_count < floor:
            sampled = class_df.sample(n=floor, replace=True, random_state=seed)
            action  = "OVERSAMPLE"
            n_over += 1
        elif real_count > ceiling:
            sampled = class_df.sample(n=ceiling, replace=False, random_state=seed)
            action  = "UNDERSAMPLE"
            n_under += 1
        else:
            sampled = class_df
            action  = "keep"
            n_keep += 1

        frames.append(sampled)
        log.info(f"  {cls:<42} {real_count:>8,} {len(sampled):>8,}  {action}")

    result = pd.concat(frames, ignore_index=True)
    result = result.reset_index(drop=True)

    log.info("-" * 72)
    log.info(f"Oversampled: {n_over}  Undersampled: {n_under}  Kept: {n_keep}")
    log.info(f"Final balanced train size: {len(result):,} rows")

    very_rare = [(c, cnt) for c, cnt in class_counts.items() if cnt < 100]
    if very_rare:
        log.warning(
            f"Very rare classes heavily oversampled (<100 real rows): "
            f"{[c for c, _ in very_rare]}. Interpret their recall cautiously."
        )
    return result

"""
label_mapping.py
================
Official CICIoT2023 label mappings for all three classification tasks.

Split strategy: split ONCE on fine-grained (34-class) labels,
then relabel the same rows for binary and 8-class tasks.
"""

import pandas as pd

# ── All 33 attack labels (as they appear in the dataset Label column) ──────────
ALL_ATTACK_LABELS = [
    "DDOS-ACK_FRAGMENTATION",
    "DDOS-UDP_FLOOD",
    "DDOS-SLOWLORIS",
    "DDOS-ICMP_FLOOD",
    "DDOS-RSTFINFLOOD",
    "DDOS-PSHACK_FLOOD",
    "DDOS-HTTP_FLOOD",
    "DDOS-UDP_FRAGMENTATION",
    "DDOS-TCP_FLOOD",
    "DDOS-SYN_FLOOD",
    "DDOS-SYNONYMOUSIP_FLOOD",
    "DDOS-ICMP_FRAGMENTATION",
    "DOS-TCP_FLOOD",
    "DOS-HTTP_FLOOD",
    "DOS-SYN_FLOOD",
    "DOS-UDP_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-GREIP_FLOOD",
    "MIRAI-UDPPLAIN",
    "RECON-HOSTDISCOVERY",
    "RECON-OSSCAN",
    "RECON-PINGSWEEP",
    "RECON-PORTSCAN",
    "VULNERABILITYSCAN",
    "DNS_SPOOFING",
    "MITM-ARPSPOOFING",
    "XSS",
    "SQLINJECTION",
    "COMMANDINJECTION",
    "BROWSERHIJACKING",
    "UPLOADING_ATTACK",
    "BACKDOOR_MALWARE",
    "DICTIONARYBRUTEFORCE",
]

ALL_LABELS = ["BENIGN"] + ALL_ATTACK_LABELS   # 34 total

# ── Binary mapping (0 = Benign, 1 = Malicious) ────────────────────────────────
BINARY_MAP: dict[str, int] = {label: 1 for label in ALL_ATTACK_LABELS}
BINARY_MAP["BENIGN"] = 0

BINARY_CLASS_NAMES = ["Benign", "Malicious"]

# ── 8-class (category) mapping ────────────────────────────────────────────────
CATEGORY_MAP: dict[str, int] = {
    # 0 — Benign
    "BENIGN": 0,
    # 1 — DDoS
    "DDOS-ACK_FRAGMENTATION": 1,
    "DDOS-UDP_FLOOD": 1,
    "DDOS-SLOWLORIS": 1,
    "DDOS-ICMP_FLOOD": 1,
    "DDOS-RSTFINFLOOD": 1,
    "DDOS-PSHACK_FLOOD": 1,
    "DDOS-HTTP_FLOOD": 1,
    "DDOS-UDP_FRAGMENTATION": 1,
    "DDOS-TCP_FLOOD": 1,
    "DDOS-SYN_FLOOD": 1,
    "DDOS-SYNONYMOUSIP_FLOOD": 1,
    "DDOS-ICMP_FRAGMENTATION": 1,
    # 2 — DoS
    "DOS-TCP_FLOOD": 2,
    "DOS-HTTP_FLOOD": 2,
    "DOS-SYN_FLOOD": 2,
    "DOS-UDP_FLOOD": 2,
    # 3 — Mirai
    "MIRAI-GREETH_FLOOD": 3,
    "MIRAI-GREIP_FLOOD": 3,
    "MIRAI-UDPPLAIN": 3,
    # 4 — Recon
    "RECON-HOSTDISCOVERY": 4,
    "RECON-OSSCAN": 4,
    "RECON-PINGSWEEP": 4,
    "RECON-PORTSCAN": 4,
    "VULNERABILITYSCAN": 4,
    # 5 — Spoofing
    "DNS_SPOOFING": 5,
    "MITM-ARPSPOOFING": 5,
    # 6 — Web
    "XSS": 6,
    "SQLINJECTION": 6,
    "COMMANDINJECTION": 6,
    "BROWSERHIJACKING": 6,
    "UPLOADING_ATTACK": 6,
    "BACKDOOR_MALWARE": 6,
    # 7 — BruteForce
    "DICTIONARYBRUTEFORCE": 7,
}

CATEGORY_CLASS_NAMES = [
    "Benign", "DDoS", "DoS", "Mirai",
    "Recon", "Spoofing", "Web", "BruteForce"
]

# ── Fine-grained mapping (34 classes, alphabetically sorted for consistency) ──
FINE_CLASS_NAMES = sorted(ALL_LABELS)          # sorted list of 34 labels
FINE_MAP: dict[str, int] = {
    label: idx for idx, label in enumerate(FINE_CLASS_NAMES)
}


# ── Helper functions ──────────────────────────────────────────────────────────

def apply_binary(label_series: pd.Series) -> pd.Series:
    """Map raw labels to binary (0/1)."""
    return label_series.map(BINARY_MAP)


def apply_category(label_series: pd.Series) -> pd.Series:
    """Map raw labels to 8-class integer codes."""
    return label_series.map(CATEGORY_MAP)


def apply_fine(label_series: pd.Series) -> pd.Series:
    """Map raw labels to 34-class integer codes."""
    return label_series.map(FINE_MAP)


def add_all_label_columns(df: pd.DataFrame, raw_col: str = "label") -> pd.DataFrame:
    """
    Add label_binary, label_category, label_fine columns to dataframe.
    The raw label column must contain strings matching the official taxonomy.
    This is the single point where all three label sets are created.
    """
    df = df.copy()
    df["label_binary"] = apply_binary(df[raw_col])
    df["label_category"] = apply_category(df[raw_col])
    df["label_fine"] = apply_fine(df[raw_col])
    return df


def get_class_names(task: str) -> list[str]:
    """Return class name list for a given task string."""
    mapping = {
        "binary": BINARY_CLASS_NAMES,
        "8class": CATEGORY_CLASS_NAMES,
        "34class": FINE_CLASS_NAMES,
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(mapping.keys())}")
    return mapping[task]


def get_label_col(task: str) -> str:
    mapping = {
        "binary": "label_binary",
        "8class": "label_category",
        "34class": "label_fine",
    }
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'.")
    return mapping[task]

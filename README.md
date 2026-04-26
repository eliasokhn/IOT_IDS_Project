# IoT Intrusion Detection System (IDS)

ML-based intrusion detection for IoT network traffic, trained on the **CICIoT2023** dataset
(University of New Brunswick — ~45 million records, 34 attack classes).

Deployed as a **FastAPI REST service** packaged in Docker. Pre-trained models are included
— no dataset download required to run inference.

---

## Docker — Quickest way to run

```bash
# Pull and run the pre-built image
docker pull alimssw/iot-ids:latest
docker run -p 8000:8000 alimssw/iot-ids:latest

# Open the interactive API docs
# http://localhost:8000/docs
```

**Test with a single request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "flow_duration": 1.23,
      "total_fwd_packets": 10.0,
      "total_bwd_packets": 8.0,
      "total_length_fwd_packets": 1500.0,
      "total_length_bwd_packets": 900.0,
      "flow_bytes_per_sec": 2400.5,
      "flow_packets_per_sec": 14.6,
      "fwd_packet_length_mean": 150.0,
      "bwd_packet_length_mean": 112.5,
      "syn_flag_count": 1.0
    }
  }'
```

**Switch model or task at runtime:**
```bash
# Gradient Boosting, 8-class attack families
docker run -p 8000:8000 -e MODEL_NAME=gb -e TASK=8class amoussawi/iot-ids:latest

# Logistic Regression, all 34 fine-grained classes
docker run -p 8000:8000 -e MODEL_NAME=lr -e TASK=34class amoussawi/iot-ids:latest
```

Available `MODEL_NAME`: `gb` (Gradient Boosting), `lr` (Logistic Regression)  
Available `TASK`: `binary`, `8class`, `34class`

---

## What this project does

- Classifies IoT network flows as **Benign** or one of **33 attack types**
- Three classification granularities in one pipeline:
  - **Binary** — Benign vs. Malicious (2 classes)
  - **8-class** — Benign + 7 attack families (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
  - **34-class** — Benign + all 33 individual attack subtypes
- Two models trained per task (6 total): **Logistic Regression** (baseline) and **Gradient Boosting** (XGBoost)
- **Streaming simulation** with real-time drift monitoring across micro-batches
- Deployable **FastAPI inference service** packaged in Docker

---

## Results

**Primary metric: Macro F1** — weights all classes equally, essential for heavily imbalanced security data.
Evaluated on a held-out 15% test set (~1.33M samples) from the CICIoT2023 dataset.

### Binary (Benign vs. Malicious)

| Model | Accuracy | Macro F1 | ROC-AUC | FPR (Benign) |
|-------|----------|----------|---------|--------------|
| Logistic Regression | 95.5% | 0.828 | 0.9896 | 0.06% |
| **Gradient Boosting** | **97.4%** | **0.884** | **0.9965** | **0.12%** |

### 8-Class (Attack Families)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 70.9% | 0.529 |
| **Gradient Boosting** | **79.0%** | **0.667** |

Top performing classes (GB): Mirai (F1=1.00), Spoofing (0.89), DDoS (0.81)  
Hardest classes: Web (0.19), BruteForce (0.21) — severely underrepresented

### 34-Class (Fine-Grained Attack Types)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 69.2% | 0.473 |
| **Gradient Boosting** | **78.5%** | **0.596** |

Top performing classes (GB): DDOS-ICMP_FLOOD (F1=1.00), MIRAI-GREETH/GREIP/UDPPLAIN (~1.00), DDOS-PSHACK_FLOOD (~1.00)  
Hardest classes: Ultra-rare attacks (<200 samples) — RECON-PINGSWEEP, XSS, UPLOADING_ATTACK

> **Note on rare classes:** The CICIoT2023 dataset is severely imbalanced (BENIGN ~862K rows, XSS ~189 rows,
> 4500× ratio). All models use `class_weight='balanced'` + BENIGN weight boost to improve fairness across classes.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check — always 200 if server is up |
| `/info` | GET | Model info, class names, feature list |
| `/classes` | GET | List class names for the loaded task |
| `/predict` | POST | Classify one traffic record |
| `/predict_batch` | POST | Classify up to 10,000 records |
| `/docs` | GET | Interactive Swagger UI |

### Single prediction response

```json
{
  "predicted_class": 1,
  "predicted_label": "Malicious",
  "probabilities": {"Benign": 0.02, "Malicious": 0.98},
  "is_malicious": true,
  "model": "gb",
  "task": "binary",
  "latency_ms": 4.2
}
```

### Batch prediction response

```json
{
  "predictions": [...],
  "n_records": 100,
  "n_malicious": 12,
  "alert_rate": 0.12,
  "latency_ms": 18.5,
  "model": "gb",
  "task": "binary"
}
```

---

## Local Setup (Without Docker)

```bash
# 1. Clone
git clone https://github.com/YOUR_ORG/iot-ids-project.git
cd iot-ids-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API (pre-trained models are included)
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

# Visit http://localhost:8000/docs
```

---

## Running the Full Pipeline (Requires Dataset)

The CICIoT2023 dataset (~45 GB) must be downloaded separately.

**Download:** https://www.unb.ca/cic/datasets/iotdataset-2023.html  
**Place all 63 CSV files in:** `data/raw/`

```bash
# First run — merges all 63 CSVs into parquet, trains all 6 models
python run_pipeline.py --real-data

# Re-run with a different sample fraction (parquet already built, reused)
python run_pipeline.py --real-data --sample-frac 0.20

# Train only specific models/tasks
python run_pipeline.py --real-data --models gb --tasks binary 8class

# Skip training, just re-evaluate saved models
python run_pipeline.py --skip-train
```

---

## Build Docker Image Locally

```bash
# Build
docker build -t iot-ids:latest .

# Run
docker run -p 8000:8000 iot-ids:latest

# Tag and push to Docker Hub (replace YOUR_USERNAME)
docker login
docker tag iot-ids:latest YOUR_USERNAME/iot-ids:latest
docker push YOUR_USERNAME/iot-ids:latest
```

---

## Repository Structure

```
iot-ids-project/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← Docker image definition
├── run_pipeline.py              ← One-command full pipeline runner
│
├── configs/
│   ├── data_config.yaml         ← Dataset paths, sample fraction, feature columns
│   ├── model_config.yaml        ← LR/GB hyperparameters, GPU settings
│   └── monitoring_config.yaml   ← Drift thresholds and batch sizes
│
├── src/
│   ├── data/                    ← CSV→Parquet loader, label mapping, splitter
│   ├── features/                ← Dedup, one-hot encoding, RobustScaler, PowerTransformer
│   ├── models/                  ← LR (cuML/sklearn) and GB (XGBoost/LightGBM/sklearn)
│   ├── evaluation/              ← Macro F1, confusion matrices, recall heatmaps
│   ├── serving/                 ← FastAPI app and Predictor inference class
│   └── monitoring/              ← Z-score drift detection, streaming simulation
│
├── models/                      ← Pre-trained artifacts (6 models + preprocessor)
│   ├── preprocessor.pkl         ← Fitted RobustScaler + PowerTransformer pipeline
│   ├── lr_binary.pkl            ← Logistic Regression — binary task
│   ├── lr_8class.pkl            ← Logistic Regression — 8-class task
│   ├── lr_34class.pkl           ← Logistic Regression — 34-class task
│   ├── gb_binary.pkl            ← Gradient Boosting (XGBoost) — binary task
│   ├── gb_8class.pkl            ← Gradient Boosting — 8-class task
│   └── gb_34class.pkl           ← Gradient Boosting — 34-class task
│
├── reports/                     ← Metrics JSON, confusion matrix PNGs
├── notebooks/                   ← Step-by-step pipeline notebooks (01–06)
└── data/
    ├── raw/                     ← 63 Merged*.csv files (NOT in git — download separately)
    └── processed/               ← Generated parquet and split files (NOT in git)
```

---

## Dataset

**CICIoT2023** — University of New Brunswick  
- ~45 million network flow records
- 1 benign class + 33 attack subtypes across 7 attack families
- 39 numeric features + Protocol Type
- Download: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Reference paper: https://www.mdpi.com/1424-8220/23/13/5941

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Deduplicate **before** splitting | ~5% of rows are exact duplicates across files. Splitting first would leak test rows into training. |
| Split **once** on 34-class labels | Reused for binary and 8-class to prevent row-level leakage across tasks. |
| RobustScaler + PowerTransformer | Variance column has values up to 46M — StandardScaler would produce near-infinite scaled values. RobustScaler uses median+IQR; PowerTransformer normalizes heavy-tailed features for LR. |
| `class_weight='balanced'` everywhere | 4500× imbalance ratio. Macro F1 requires good recall on rare classes. |
| Protocol Type one-hot encoded | Protocol numbers {0,1,6,17,47} are nominal (IP protocol IDs), not ordinal. Treating them as numeric implies false ordering. |

---

## References

- CICIoT2023 dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- CICIoT2023 paper: https://www.mdpi.com/1424-8220/23/13/5941

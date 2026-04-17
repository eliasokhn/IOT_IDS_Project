# Streaming IoT Intrusion Detection with Drift Monitoring

A university final project that detects malicious IoT network traffic using machine learning.
Trained on the **CICIoT2023** dataset (~45M records, 34 classes).

---

## What this project does

- Classifies IoT network flows as **Benign** or one of **33 attack types**
- Three classification levels: binary (2), family (8), fine-grained (34)
- Two models: **Logistic Regression** (baseline) vs **Gradient Boosting** (stronger)
- **Streaming simulation** with real-time drift monitoring across micro-batches
- Deployable **FastAPI inference service** packaged in Docker

---

## Dataset

**CICIoT2023** — University of New Brunswick  
Download: https://www.unb.ca/cic/datasets/iotdataset-2023.html  
Place all CSV files inside `data/raw/`.

~45 million network flow records — 1 benign class + 33 attack types.

---

## Quick Start (Without Real Dataset)

The project includes a **synthetic demo mode** that mirrors the real dataset structure.
You can run the full pipeline immediately without downloading anything.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_ORG/iot-ids-project.git
cd iot-ids-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (uses synthetic demo data)
python run_pipeline.py

# 4. Start the API
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs
```

---

## Full Setup (With Real Dataset)

```bash
# 1. Download CICIoT2023 CSVs → place in data/raw/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order (in Google Colab or locally):
#    notebooks/01_data_exploration.py
#    notebooks/02_preprocessing.py
#    notebooks/03_train_logistic.py
#    notebooks/04_train_gradient_boost.py
#    notebooks/05_evaluation.py
#    notebooks/06_streaming_demo.py

# In each notebook, set USE_DEMO = False to use the real dataset.
```

---

## Running Notebooks in Google Colab

```python
# Cell 1: Clone and set up
!git clone https://github.com/YOUR_ORG/iot-ids-project.git
%cd iot-ids-project
!pip install -r requirements.txt

# Cell 2: Mount Drive (if data is on Drive)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Run the notebook script
%run notebooks/01_data_exploration.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests pass without the real dataset or trained models.

---

## API Usage

### Start the server

```bash
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "flow_duration": 1.23,
      "total_fwd_packets": 10.0,
      "total_bwd_packets": 8.0,
      "flow_bytes_per_sec": 2400.5,
      "syn_flag_count": 1.0
    }
  }'
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"flow_duration": 1.2, "total_fwd_packets": 5.0}, ...]}'
```

### Health check

```bash
curl http://localhost:8000/health
```

Visit **http://localhost:8000/docs** for the full interactive Swagger UI.

---

## Docker

### Pull and run the pre-built image

```bash
docker pull YOUR_DOCKERHUB_USERNAME/iot-ids:latest
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/iot-ids:latest
```

### Build locally

```bash
docker build -t iot-ids:latest .
docker run -p 8000:8000 iot-ids:latest
```

### Change model or task at runtime

```bash
# Use 8-class GB model
docker run -p 8000:8000 -e MODEL_NAME=gb -e TASK=8class iot-ids:latest

# Use LR binary model
docker run -p 8000:8000 -e MODEL_NAME=lr -e TASK=binary iot-ids:latest
```

### Push to Docker Hub

```bash
docker tag iot-ids:latest YOUR_DOCKERHUB_USERNAME/iot-ids:latest
docker push YOUR_DOCKERHUB_USERNAME/iot-ids:latest
```

---

## Repository Structure

```
iot-ids-project/
├── README.md                    ← This file
├── architecture.md              ← System architecture
├── requirements.txt
├── Dockerfile
├── run_pipeline.py              ← One-command full pipeline runner
│
├── configs/
│   ├── data_config.yaml         ← Dataset paths and sampling config
│   ├── model_config.yaml        ← Hyperparameters for both models
│   └── monitoring_config.yaml   ← Drift thresholds and batch sizes
│
├── data/
│   ├── README.md                ← Download instructions
│   ├── raw/                     ← Raw CSVs (not committed)
│   └── processed/               ← Parquet, split indices (not committed)
│
├── notebooks/                   ← Run in order (Colab or local)
│   ├── 01_data_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_train_logistic.py
│   ├── 04_train_gradient_boost.py
│   ├── 05_evaluation.py
│   └── 06_streaming_demo.py
│
├── src/
│   ├── data/                    ← Loader, label mapping, splitter, validation
│   ├── features/                ← Preprocessor (scaler + imputer)
│   ├── models/                  ← LR and GB training scripts
│   ├── evaluation/              ← Metrics, confusion matrices, plots
│   ├── serving/                 ← FastAPI app and Predictor class
│   └── monitoring/              ← DriftMonitor and streaming simulation
│
├── models/                      ← Trained .pkl artifacts (after training)
├── reports/                     ← Metrics JSON, plots PNG, monitoring logs
├── slides/                      ← Final presentation
├── docs/                        ← One-page report
└── tests/                       ← Pytest unit tests
```

---

## Results Summary

After running the full pipeline, results are saved in `reports/model_comparison.csv`.

**Primary metric: Macro F1** (treats all 34 classes equally, critical for imbalanced security data)

| Task | LR Macro F1 | GB Macro F1 | Winner |
|------|-------------|-------------|--------|
| Binary (2-class) | — | — | — |
| Family (8-class) | — | — | — |
| Fine-grained (34-class) | — | — | — |

*Fill in after running the training notebooks.*

---

## Team

| Member | Role |
|--------|------|
| Member 1 | Data & Preprocessing |
| Member 2 | Model Training & Evaluation |
| Member 3 | Deployment & Docker |
| Member 4 | Documentation & Presentation |

---

## References

- CICIoT2023 dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- CICIoT2023 paper: https://www.mdpi.com/1424-8220/23/13/5941
- Deadline: Monday, April 20th at 12 PM noon

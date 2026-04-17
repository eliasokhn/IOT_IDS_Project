# One-Page Project Report
## Streaming IoT Intrusion Detection with Drift Monitoring

---

**Group Members:**
- [Student Name 1] — [USJ Email]
- [Student Name 2] — [USJ Email]
- [Student Name 3] — [USJ Email]
- [Student Name 4] — [USJ Email]

**Course:** Machine Learning Final Project — USJ
**Submission Date:** Monday, April 20, 2026

---

## Data Source

**CICIoT2023** — Canadian Institute for Cybersecurity, University of New Brunswick
- URL: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- ~45 million network flow records from a realistic IoT environment
- 34 classes: 1 benign class + 33 attack types across 7 attack families
- Heavy class imbalance: benign ~1M rows, BruteForce only ~12K rows

---

## Approach

We built a full ML pipeline for intrusion detection on IoT network traffic.

**Split strategy:** One stratified 70/15/15 split on 34-class labels, then relabeled three times (binary, 8-class, 34-class). This guarantees fair, leak-free comparison across all tasks.

**Preprocessing:** StandardScaler fitted on training data only. Infinite values replaced with column maxima. Missing values imputed with column medians.

**Three classification tasks:**
- Binary: Benign vs. Malicious
- 8-class: Benign + 7 attack families (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
- 34-class: All fine-grained labels

---

## Methods Used

| Component | Method |
|-----------|--------|
| Baseline model | Logistic Regression (class_weight=balanced, lbfgs solver) |
| Strong model | HistGradientBoostingClassifier (early stopping, class_weight=balanced) |
| Imbalance handling | class_weight='balanced' in both models |
| Primary metric | Macro F1 (treats all classes equally) |
| Serving | FastAPI REST API, Docker image on Docker Hub |
| Drift detection | Z-score on top-5 features + alert rate monitoring |
| Streaming | Micro-batches of 500 records, 3 scenarios tested |

---

## Summary of Results

| Task | LR Macro F1 | GB Macro F1 | FPR Benign (GB) |
|------|-------------|-------------|-----------------|
| Binary (2-class) | — | — | — |
| Family (8-class) | — | — | — |
| Fine-grained (34-class) | — | — | — |

*Fill in after running the training notebooks.*

**Key finding:** Gradient Boosting outperforms Logistic Regression on all three tasks, with the largest gap on the 34-class fine-grained task. Rare attack classes (Web, BruteForce) show low recall in both models, highlighting the challenge of class imbalance in intrusion detection.

**Streaming:** The drift monitor correctly detected alert rate spikes in the attack surge scenario (batch 20–30) and gradually increasing drift in the gradual drift scenario.

---

## GitHub Repository

https://github.com/YOUR_ORG/iot-ids-project

**Docker Hub Image:** `YOUR_DOCKERHUB_USERNAME/iot-ids:latest`

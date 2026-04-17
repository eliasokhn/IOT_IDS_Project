# Architecture — Streaming IoT Intrusion Detection with Drift Monitoring

---

## 1. Overall System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
│                                                                             │
│  CICIoT2023 CSVs                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  build_merged_parquet()  →  data/processed/merged.parquet                  │
│       │                                                                     │
│       ▼                                                                     │
│  validate_dataset()      →  schema checks, missing values, class counts     │
│       │                                                                     │
│       ▼                                                                     │
│  sample_stratified()     →  10% per class (for Colab memory safety)         │
│       │                                                                     │
│       ▼                                                                     │
│  add_all_label_columns() →  label_binary | label_category | label_fine      │
│       │                                                                     │
│       ▼                                                                     │
│  create_splits()         →  ONE stratified 70/15/15 split on label_fine     │
│       │                          (same rows reused for all 3 tasks)         │
│       ▼                                                                     │
│  Preprocessor.fit()      →  fit StandardScaler + imputer on TRAIN only      │
│       │                                                                     │
│       ▼                                                                     │
│  Preprocessor.transform()→  apply to train / val / test                     │
│       │                                                                     │
│  ┌────┴─────────────────────────────────────────┐                           │
│  │  6 Training Experiments (2 models × 3 tasks) │                           │
│  │                                              │                           │
│  │  LR binary    LR 8class    LR 34class        │                           │
│  │  GB binary    GB 8class    GB 34class        │                           │
│  └────┬─────────────────────────────────────────┘                           │
│       │                                                                     │
│       ▼                                                                     │
│  Evaluate on TEST set  →  macro F1, per-class recall, FPR, confusion matrix │
│       │                                                                     │
│       ▼                                                                     │
│  Save artifacts:                                                            │
│    models/preprocessor.pkl                                                  │
│    models/lr_binary.pkl  ...  models/gb_34class.pkl                         │
│    models/class_names_*.json                                                │
│    reports/metrics_*.json                                                   │
│    reports/cm_*.png                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT PIPELINE                                  │
│                                                                             │
│  Docker Image                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  src/serving/api.py  (FastAPI)                                     │     │
│  │       │                                                            │     │
│  │       ▼                                                            │     │
│  │  Predictor.load()   ←  models/gb_binary.pkl + preprocessor.pkl    │     │
│  │       │                                                            │     │
│  │  POST /predict      →  preprocess → predict → return JSON          │     │
│  │  POST /predict_batch→  batch predict → alert rate → return JSON    │     │
│  │  GET  /health       →  {"status": "ok"}                            │     │
│  │  GET  /info         →  model name, task, class names               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Client sends:  {"features": {"flow_duration": 1.2, ...}}                  │
│  API returns:   {"predicted_label": "Benign", "is_malicious": false, ...}  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        MONITORING PIPELINE                                  │
│                                                                             │
│  Test set (held-out, never used in training)                                │
│       │                                                                     │
│       ▼  (micro-batches of 500 records)                                     │
│  run_simulation()                                                           │
│       │                                                                     │
│       ├── predict each batch  (via Predictor)                               │
│       │                                                                     │
│       ├── DriftMonitor.check_batch()                                        │
│       │       ├── Z-score on top-5 features  →  FEATURE DRIFT warning       │
│       │       ├── alert_rate tracking        →  HIGH ALERT RATE warning     │
│       │       └── prediction distribution   →  stored in history            │
│       │                                                                     │
│       ▼                                                                     │
│  3 Scenarios:                                                               │
│    1. Normal traffic       → baseline behavior                              │
│    2. Attack surge         → sudden spike detected at batch 20              │
│    3. Gradual drift        → slow increase, caught by Z-score monitor       │
│       │                                                                     │
│       ▼                                                                     │
│  reports/monitoring_logs/*.json                                             │
│  reports/monitoring_plots/*.png                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

```
Raw CSVs (data/raw/)
    │
    │  build_merged_parquet()
    ▼
merged.parquet   ← single file, all labels uppercase, columns normalised
    │
    │  load_dataset(sample_frac=0.10)
    ▼
df_raw (N × F)   ← 10% per class, ~4.5M rows total
    │
    │  add_all_label_columns()
    ▼
df with columns:
  [feature_0 ... feature_F | label | label_binary | label_category | label_fine]
    │
    │  create_splits(stratify_col="label_fine")   ← SPLIT ONCE
    ▼
train_df (70%)  val_df (15%)  test_df (15%)
    │
    │  get_X_y(task)   ← called three times with different task strings
    ▼
(X_train, y_train_binary)     (X_train, y_train_8class)     (X_train, y_train_34class)
(X_val,   y_val_binary)       (X_val,   y_val_8class)       (X_val,   y_val_34class)
(X_test,  y_test_binary)      (X_test,  y_test_8class)      (X_test,  y_test_34class)
```

### Why we split once and relabel three times

If you split separately for each task, the same physical row (say, a DDOS-TCP_FLOOD record) could end up in training for the binary task but in the test set for the 34-class task. That means the model has already seen that row at one level of granularity. This leaks information and makes cross-task comparison unfair.

By splitting once on the finest labels (34-class) and relabeling, every row has a permanent, fixed assignment. The test set is clean at all granularity levels simultaneously.

---

## 3. Label Mapping

```
Raw label string (e.g., "DDOS-TCP_FLOOD")
    │
    ├──  apply_binary()    →  1  (Malicious)
    ├──  apply_category()  →  1  (DDoS)
    └──  apply_fine()      →  8  (sorted alphabetical index)

All three mappings live in src/data/label_mapping.py
They are deterministic dictionaries — no model training involved.
```

---

## 4. Preprocessing Flow

```
X_train (raw DataFrame)
    │
    │  Preprocessor.fit(X_train)
    │    ├── compute col_maxes  (max of finite values per column)
    │    ├── replace inf → col_max
    │    ├── compute col_medians
    │    └── StandardScaler.fit()
    │
    ▼  (saved as models/preprocessor.pkl)

Preprocessor.transform(X_train)  →  X_train_scaled  (fit on train)
Preprocessor.transform(X_val)    →  X_val_scaled    (apply train stats)
Preprocessor.transform(X_test)   →  X_test_scaled   (apply train stats)

At inference time:
  Preprocessor.load("models/preprocessor.pkl")
  preprocessor.transform(new_record)              ← EXACT same logic
```

**Critical rule:** The scaler is fit on training data ONLY. Validation and test data are transformed using training statistics. This is what `Preprocessor.save()` / `Preprocessor.load()` ensures at inference time.

---

## 5. Model Architecture

### Logistic Regression (Baseline)

```
X_scaled  →  Linear combination  →  Softmax  →  Class probabilities
             (W × features + b)

Config: C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs'
Handles multi-class via multinomial softmax.
Full-batch training (loads all training data at once).
```

### Gradient Boosting (HistGradientBoostingClassifier)

```
X_scaled  →  Tree 1 (weak learner)
               │  residuals
               ▼
             Tree 2 (corrects errors of Tree 1)
               │  residuals
               ▼
             ...
               │
             Tree N  (early stopping: stops when val loss stops improving)
               │
               ▼
          Ensemble prediction  →  Class probabilities (softmax)

Config: max_iter=300, learning_rate=0.1, max_depth=6,
        class_weight='balanced', early_stopping=True, validation_fraction=0.1
Full-batch training. Mini-batches NOT used (correct for tree-based models).
```

---

## 6. Evaluation Flow

```
For each of 6 experiments (2 models × 3 tasks):

  model.predict(X_test_scaled)         →  y_pred  (integer class indices)
  model.predict_proba(X_test_scaled)   →  y_proba (N × n_classes matrix)

  compute_all_metrics(y_test, y_pred, y_proba, class_names):
    ├── accuracy_score()
    ├── f1_score(average='macro')         ← PRIMARY METRIC
    ├── f1_score(average='weighted')
    ├── precision_score(average=None)     ← per class
    ├── recall_score(average=None)        ← per class
    ├── confusion_matrix()
    ├── FPR on benign  =  benign_FP / (benign_FP + benign_TN)
    └── roc_auc_score()                   ← binary task only

  Results saved as:
    reports/metrics_{model}_{task}.json
    reports/cm_{model}_{task}.png
    reports/recall_{model}_{task}.png
```

---

## 7. Streaming Simulation Flow

```
test_df (never seen during training)
    │
    │  slice micro-batches of 500 rows
    ▼
Batch 1  →  predict_fn()  →  y_pred  →  DriftMonitor.check_batch()
Batch 2  →  predict_fn()  →  y_pred  →  DriftMonitor.check_batch()
...
Batch N  →  predict_fn()  →  y_pred  →  DriftMonitor.check_batch()
    │
    ▼
Per-batch log:
  ├── alert_rate    (fraction of batch predicted malicious)
  ├── z_score       (per top-5 feature)
  ├── drift_warnings (list of triggered alerts)
  └── pred_distribution (class counts)

Scenarios:
  1. Normal:       no injection, baseline behavior
  2. Attack Surge: 70% of batch replaced with attack records at batch 20
  3. Gradual Drift: attack fraction grows 5% per batch from batch 10

Output:
  reports/monitoring_plots/alert_rate_*.png
  reports/monitoring_plots/feature_drift_*.png
  reports/monitoring_logs/monitoring_log_*.json
```

---

## 8. Tool Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data | pandas, pyarrow | Load, merge, process CSVs and Parquet |
| ML | scikit-learn | LR, GBM, preprocessing, metrics |
| Serving | FastAPI, uvicorn | REST API for inference |
| Packaging | Docker | Reproducible deployment |
| Registry | Docker Hub | Public image for submission |
| Experiment | Google Colab | GPU/TPU cloud for training notebooks |
| Development | VS Code | Modules, API, Docker, configs |
| Version control | GitHub | Code, slides, report |
| Testing | pytest | Unit tests for core modules |

---

## 9. File Ownership Map

| Files | Team Member |
|-------|-------------|
| `src/data/`, `src/features/`, `notebooks/01`, `notebooks/02` | Member 1 — Data |
| `src/models/`, `src/evaluation/`, `notebooks/03-05` | Member 2 — Models |
| `src/serving/`, `src/monitoring/`, `Dockerfile`, `notebooks/06` | Member 3 — Deploy |
| `README.md`, `architecture.md`, `docs/`, `slides/`, `tests/` | Member 4 — Docs |

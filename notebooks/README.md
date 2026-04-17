# Notebooks

Run these notebooks **in order**. Each one saves outputs for the next.

## Execution Order

| # | Notebook | What it does | Runtime (demo) |
|---|----------|--------------|----------------|
| 01 | `01_data_exploration.py` | Load data, validate, plot class distribution | ~1 min |
| 02 | `02_preprocessing.py` | Create splits, fit scaler, save all arrays | ~1 min |
| 03 | `03_train_logistic.py` | Train LR for binary / 8-class / 34-class | ~3 min |
| 04 | `04_train_gradient_boost.py` | Train GB for binary / 8-class / 34-class | ~5 min |
| 05 | `05_evaluation.py` | Compare models, generate report plots | ~1 min |
| 06 | `06_streaming_demo.py` | Streaming simulation + drift monitoring | ~2 min |

## How to Run in Google Colab

```python
# Cell 1 — Clone the repo
!git clone https://github.com/YOUR_ORG/iot-ids-project.git
%cd iot-ids-project
!pip install -r requirements.txt

# Cell 2 — Run a notebook (example: notebook 01)
%run notebooks/01_data_exploration.py
```

## How to Run Locally

```bash
# From the project root
python notebooks/01_data_exploration.py
```

Or convert to .ipynb with:
```bash
jupytext --to notebook notebooks/01_data_exploration.py
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Demo vs Real Data

Each notebook has a flag at the top:
```python
USE_DEMO = True   # ← set to False for real CICIoT2023 data
```

## Important Rules

1. Run notebooks **in order** — each saves files needed by the next
2. Do NOT commit notebooks with cell outputs — clear outputs before `git push`
3. All logic lives in `src/` — notebooks only call functions and display results
4. If a notebook fails, check that the previous notebook completed successfully

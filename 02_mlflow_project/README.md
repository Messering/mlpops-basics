# MLflow Project: Wine Quality Classifier

A reproducible ML project that demonstrates MLflow Projects — entry points, environment management, and CLI parameters.

## Project Structure

```
02_mlflow_project/
├── MLproject          # Project configuration (entry points + params)
├── conda.yaml         # Conda environment specification
├── train.py           # Training script (logs to MLflow + saves model locally)
├── evaluate.py        # Evaluation script (classification report)
├── predict.py         # Prediction script (batch inference → CSV)
├── data/              # Dataset directory (created automatically)
│   └── wine.csv
└── README.md
```

## What is an MLflow Project?

An MLflow Project packages ML code so it can be reproduced anywhere:

| Component | Purpose |
|-----------|---------|
| **MLproject** | Declares entry points, parameters, and the environment |
| **conda.yaml** | Pins Python & library versions for reproducibility |
| **Scripts** | Python files with Click CLI interfaces |

## Quick Start

### Prerequisites

```bash
pip install mlflow scikit-learn pandas matplotlib seaborn click joblib
```

### Option A — Run scripts directly

```bash
cd 02_mlflow_project

python train.py                          # train with defaults
python train.py --n-estimators 200       # custom hyper-parameters
python evaluate.py                       # evaluate the trained model
python predict.py                        # batch predictions → predictions.csv
```

### Option B — Run via `mlflow run`

```bash
# Use --env-manager=local to skip creating a new conda env
mlflow run . -e main     --env-manager=local
mlflow run . -e evaluate --env-manager=local
mlflow run . -e predict  --env-manager=local
```

With custom parameters:

```bash
mlflow run . -e main --env-manager=local \
  -P n_estimators=200 \
  -P max_depth=10 \
  -P min_samples_split=5 \
  -P test_size=0.25
```

## Entry Points

### 1. `main` — Training

Trains a Random Forest classifier and logs everything to MLflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | path | `data/wine.csv` | Training CSV (auto-created if missing) |
| `test_size` | float | `0.2` | Fraction reserved for testing |
| `n_estimators` | int | `100` | Number of trees |
| `max_depth` | int | `5` | Maximum tree depth |
| `min_samples_split` | int | `2` | Min samples to split a node |
| `random_state` | int | `42` | Random seed |

**Outputs:** `models/model.pkl`, confusion matrix PNG, feature importance PNG, MLflow run with metrics.

### 2. `evaluate` — Evaluation

Loads the saved model and prints a classification report.

| Parameter | Type | Default |
|-----------|------|---------|
| `model_path` | path | `models/model.pkl` |
| `data_path` | path | `data/wine.csv` |

### 3. `predict` — Batch Prediction

Runs the model on input data and writes predictions to CSV.

| Parameter | Type | Default |
|-----------|------|---------|
| `model_path` | path | `models/model.pkl` |
| `input_data` | path | `data/test_input.csv` (auto-created if missing) |
| `output_path` | path | `predictions.csv` |

## Viewing Results

```bash
mlflow ui
```

Open http://localhost:5000 to see experiment runs, parameters, metrics, and artifacts.

## Key Concepts

### MLproject File

```yaml
name: WineQualityClassifier
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
    command: "python train.py --n-estimators {n_estimators}"
```

### Environment Management

`conda.yaml` pins the exact versions so anyone can reproduce the environment:

```yaml
dependencies:
  - python=3.12
  - scikit-learn>=1.5
  - mlflow>=3.0
```

### CLI with Click

Each script uses Click for clean argument parsing:

```python
@click.command()
@click.option("--n-estimators", type=int, default=100)
def train(n_estimators):
    ...
```

## Running from a Git Repository

```bash
mlflow run https://github.com/<user>/<repo> -e main -P n_estimators=200
```

## Exercise

1. Run the project with default parameters
2. Try different values for `n_estimators` and `max_depth`
3. Compare runs in the MLflow UI
4. Evaluate the best model with the `evaluate` entry point
5. Create custom input data and generate predictions

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Conda env not found | Use `--env-manager=local` or install conda |
| Module not found | Activate the right virtualenv / conda env first |
| Model file not found | Run `train.py` before `evaluate.py` / `predict.py` |

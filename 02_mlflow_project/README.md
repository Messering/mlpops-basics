# MLflow Project Example: Wine Quality Classifier

This MLflow Project demonstrates how to structure a reproducible machine learning project with proper entry points, environment management, and CLI parameters.

## 📋 Project Structure

```
mlflow_project_example/
├── MLproject              # MLflow project configuration
├── conda.yaml            # Conda environment specification
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── predict.py            # Prediction script
├── data/                 # Data directory (created automatically)
└── README.md            # This file
```

## 🎯 What is an MLflow Project?

An MLflow Project is a format for packaging data science code in a reusable and reproducible way. It includes:

- **MLproject file**: Defines entry points, parameters, and environment
- **Environment specification**: `conda.yaml` or `requirements.txt`
- **Code**: Python scripts with CLI interfaces

## 🚀 Running the Project

### Prerequisites

Make sure you have MLflow and conda installed:
```bash
pip install mlflow
```

### 1. Training the Model (Main Entry Point)

Run the training script with default parameters:
```bash
mlflow run . -e main
```

Run with custom parameters:
```bash
mlflow run . -e main -P n_estimators=200 -P max_depth=10
```

All available parameters:
```bash
mlflow run . -e main \
  -P n_estimators=150 \
  -P max_depth=8 \
  -P min_samples_split=5 \
  -P test_size=0.25 \
  -P random_state=42
```

### 2. Evaluating the Model

Evaluate a trained model:
```bash
mlflow run . -e evaluate -P model_path=models/model.pkl
```

Evaluate from MLflow Model Registry:
```bash
mlflow run . -e evaluate -P model_name=WineClassifier -P model_version=1
```

### 3. Making Predictions

Make predictions with the trained model:
```bash
mlflow run . -e predict -P model_path=models/model.pkl
```

Custom input and output:
```bash
mlflow run . -e predict \
  -P model_path=models/model.pkl \
  -P input_data=data/test_input.csv \
  -P output_path=my_predictions.csv
```

## 📊 Understanding the Entry Points

### 1. `main` (Training)
- **Purpose**: Train a Random Forest classifier
- **Parameters**: 
  - `n_estimators` (int): Number of trees
  - `max_depth` (int): Maximum tree depth
  - `min_samples_split` (int): Minimum samples to split
  - `test_size` (float): Test set proportion
  - `random_state` (int): Random seed
- **Outputs**:
  - Trained model saved to `models/model.pkl`
  - MLflow run with metrics and artifacts
  - Confusion matrix and feature importance plots

### 2. `evaluate` (Evaluation)
- **Purpose**: Evaluate model performance
- **Parameters**:
  - `model_path` (str): Path to saved model
  - `model_name` (str): Model name in registry (optional)
  - `model_version` (int): Model version (optional)
- **Outputs**:
  - Classification report
  - Performance metrics

### 3. `predict` (Prediction)
- **Purpose**: Make predictions on new data
- **Parameters**:
  - `model_path` (str): Path to saved model
  - `input_data` (str): Input CSV file
  - `output_path` (str): Output CSV file
- **Outputs**:
  - Predictions saved to CSV file

## 🎓 Teaching Points (Slide 13)

### Why Use MLflow Projects?

1. **Reproducibility**: Anyone can run your code with the same environment
2. **Parameterization**: Easy to experiment with different parameters
3. **Organization**: Clear structure with defined entry points
4. **Portability**: Can run locally, on a server, or in the cloud

### Key Concepts

#### MLproject File
```yaml
name: WineQualityClassifier

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
    command: "python train.py --n-estimators {n_estimators}"
```

#### Environment Management
The `conda.yaml` file ensures everyone uses the same package versions:
```yaml
dependencies:
  - python=3.9
  - scikit-learn=1.0.2
  - mlflow=2.8.0
```

#### CLI with Click
Each script uses Click for clean command-line interfaces:
```python
@click.command()
@click.option('--n-estimators', type=int, default=100)
def train(n_estimators):
    # Training code
```

## 🔄 Running from GitHub

MLflow Projects can be run directly from Git repositories:

```bash
mlflow run https://github.com/your-repo/mlflow-project \
  -e main -P n_estimators=200
```

## 📈 Viewing Results

After running the project, view results in MLflow UI:

```bash
mlflow ui
```

Navigate to http://localhost:5000 to see:
- Experiment runs
- Parameters used
- Metrics logged
- Artifacts (plots, models)

## 🔧 Advanced Usage

### Running with Specific Experiment

```bash
mlflow run . -e main --experiment-name my-experiment
```

### Running Specific Git Commit

```bash
mlflow run . -v <commit-hash> -e main
```

### Running on Remote Server

Configure backend and artifact stores in your environment, then:
```bash
mlflow run . -e main --backend kubernetes
```

## 📝 Exercise

1. **Basic Run**: Run the project with default parameters
2. **Parameter Tuning**: Try different values for `n_estimators` and `max_depth`
3. **Compare Runs**: Use MLflow UI to compare different runs
4. **Model Evaluation**: Use the evaluate entry point on your best model
5. **Make Predictions**: Create custom input data and generate predictions

## 🆘 Troubleshooting

### Issue: Conda environment not found
**Solution**: Make sure conda is installed and in your PATH

### Issue: Module not found errors
**Solution**: Delete `mlruns/` and `.conda/` folders, then run again

### Issue: Model file not found
**Solution**: Run the training entry point first before evaluate/predict

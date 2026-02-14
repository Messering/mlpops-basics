import click
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def save_wine_dataset(output_path="data/wine.csv"):
    """Save wine dataset to CSV if it doesn't exist yet."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return output_path


def _is_mlflow_run_active():
    """Return True when launched via `mlflow run` (env var is set)."""
    return os.environ.get("MLFLOW_RUN_ID") is not None


@click.command()
@click.option("--data-path", default="data/wine.csv", help="Path to training data")
@click.option("--test-size", default=0.2, type=float, help="Test set ratio (0-1)")
@click.option("--n-estimators", default=100, type=int, help="Number of trees")
@click.option("--max-depth", default=5, type=int, help="Maximum tree depth")
@click.option("--min-samples-split", default=2, type=int, help="Min samples to split a node")
@click.option("--random-state", default=42, type=int, help="Random seed")
def train(data_path, test_size, n_estimators, max_depth, min_samples_split, random_state):
    """Train a Random Forest classifier with MLflow tracking."""

    if not _is_mlflow_run_active():
        mlflow.set_experiment("MLflow_Project_Demo")

    with mlflow.start_run(run_name=None if _is_mlflow_run_active() else "Wine_Classifier_Training"):

        print("\n" + "=" * 60)
        print("MLflow Project — Wine Quality Classifier Training")
        print("=" * 60)

        # Create the dataset from sklearn if it is missing
        if not os.path.exists(data_path):
            print(f"\nDataset not found at {data_path}, creating it...")
            data_path = save_wine_dataset(data_path)

        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1).values
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set: {X_train.shape}")
        print(f"Test set:     {X_test.shape}")

        # Log hyper-parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
            "test_size": test_size,
        }
        mlflow.log_params(params)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Feature importance artifact
        feature_names = df.drop("target", axis=1).columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(
            range(len(importances)),
            [feature_names[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.tight_layout()
        importance_path = "feature_importance.png"
        plt.savefig(importance_path)
        mlflow.log_artifact(importance_path)
        plt.close()

        # Log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:3],
        )

        # Save model locally so evaluate.py / predict.py can use it
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        print("\nModel saved to  models/model.pkl")
        print("Artifacts logged to MLflow — run  mlflow ui  to view")


if __name__ == "__main__":
    train()

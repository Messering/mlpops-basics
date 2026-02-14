import click
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


@click.command()
@click.option("--model-path", default="models/model.pkl", help="Path to trained model")
@click.option("--data-path", default="data/wine.csv", help="Path to dataset CSV")
def evaluate(model_path, data_path):
    """Evaluate a trained model on the held-out test set."""

    print("\n" + "=" * 60)
    print("MLflow Project — Model Evaluation")
    print("=" * 60)

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1).values
    y = df["target"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Evaluation completed!")


if __name__ == "__main__":
    evaluate()

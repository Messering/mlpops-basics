import click
import pandas as pd
import joblib
import os
from sklearn.datasets import load_wine


@click.command()
@click.option("--model-path", default="models/model.pkl", help="Path to trained model")
@click.option("--input-data", default="data/test_input.csv", help="Path to input CSV")
@click.option("--output-path", default="predictions.csv", help="Path to save predictions")
def predict(model_path, input_data, output_path):
    """Make predictions using a trained model."""

    print("\n" + "=" * 60)
    print("MLflow Project — Prediction")
    print("=" * 60)

    model = joblib.load(model_path)

    if not os.path.exists(input_data):
        print(f"Input file not found at {input_data}, creating sample data...")
        wine = load_wine()
        sample_df = pd.DataFrame(wine.data[:5], columns=wine.feature_names)
        os.makedirs(os.path.dirname(input_data), exist_ok=True)
        sample_df.to_csv(input_data, index=False)

    X = pd.read_csv(input_data)

    predictions = model.predict(X)

    output_df = X.copy()
    output_df["prediction"] = predictions
    output_df.to_csv(output_path, index=False)

    print(f"\nPredictions ({len(X)} samples) saved to {output_path}")
    print(output_df.head())
    print("\nPrediction completed!")


if __name__ == "__main__":
    predict()

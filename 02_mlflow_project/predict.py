import click
import mlflow
import pandas as pd
import os


@click.command()
@click.option('--model-path', default='models/model.pkl', help='Path to trained model')
@click.option('--input-data', default='data/test_input.csv', help='Path to input data')
@click.option('--output-path', default='predictions.csv', help='Path to save predictions')
def predict(model_path, input_data, output_path):
    """Make predictions using trained model"""
    
    print("\n" + "="*60)
    print("MLflow Project - Prediction")
    print("="*60)
    
    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}")
        model = mlflow.sklearn.load_model(model_path)
    else:
        print(f"\nModel not found at {model_path}")
        print("Attempting to load from MLflow Model Registry...")
        model = mlflow.pyfunc.load_model("models:/WineClassifier/Production")
    
    print(f"Loading input data from {input_data}")
    if not os.path.exists(input_data):
        print(f"Creating sample input data at {input_data}")
        from sklearn.datasets import load_wine
        wine = load_wine()
        sample_df = pd.DataFrame(wine.data[:5], columns=wine.feature_names)
        os.makedirs(os.path.dirname(input_data), exist_ok=True)
        sample_df.to_csv(input_data, index=False)
    
    X = pd.read_csv(input_data)
    
    print(f"\nMaking predictions on {len(X)} samples...")
    predictions = model.predict(X)
    
    output_df = X.copy()
    output_df['prediction'] = predictions
    output_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    print("\nSample predictions:")
    print(output_df.head())
    
    print("\n Prediction completed successfully!")


if __name__ == '__main__':
    predict()

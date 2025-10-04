import click
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os


@click.command()
@click.option('--model-path', default='models/model.pkl', help='Path to trained model')
@click.option('--data-path', default='data/wine.csv', help='Path to test data')
def evaluate(model_path, data_path):
    """Evaluate a trained model"""
    
    print("\n" + "="*60)
    print("MLflow Project - Model Evaluation")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Use same split for evaluation
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = mlflow.sklearn.load_model(model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Attempting to load latest model from MLflow...")
        model = mlflow.sklearn.load_model("models:/WineClassifier/Production")
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n Evaluation completed successfully!")


if __name__ == '__main__':
    evaluate()

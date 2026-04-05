"""
Model Management Script for MLflow Models.

This script helps manage models between mlruns/ and models/ directories.
"""
import argparse
import mlflow
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from model_inspector import (
    inspect_model_structure,
    export_model,
    compare_models,
    get_model_size
)


def list_experiments():
    """List all experiments."""
    experiments = mlflow.search_experiments()
    
    print("\n" + "="*70)
    print("EXPERIMENTS")
    print("="*70)
    for exp in experiments:
        print(f"ID: {exp.experiment_id:20} Name: {exp.name}")
    print("="*70 + "\n")


def list_runs(experiment_id):
    """List all runs in an experiment."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    print("\n" + "="*70)
    print(f"RUNS IN EXPERIMENT {experiment_id}")
    print("="*70)
    
    if len(runs) == 0:
        print("No runs found.")
    else:
        for _, run in runs.iterrows():
            print(f"Run ID: {run['run_id']}")
            print(f"  Name: {run.get('tags.mlflow.runName', 'N/A')}")
            print(f"  Status: {run['status']}")
            print(f"  Start: {run['start_time']}")
            
            # Print metrics if available
            metric_cols = [col for col in run.index if col.startswith('metrics.')]
            if metric_cols:
                print("  Metrics:")
                for col in metric_cols:
                    metric_name = col.replace('metrics.', '')
                    print(f"    {metric_name}: {run[col]:.4f}")
            print()
    
    print("="*70 + "\n")


def inspect_model(run_id, artifact_path="model"):
    """Inspect model structure."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    print(f"\n🔍 Inspecting model: {model_uri}\n")
    inspect_model_structure(model_uri, verbose=True)


def export_model_cmd(run_id, output_dir, model_name=None, artifact_path="model"):
    """Export model to directory."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    if model_name is None:
        model_name = f"model_{run_id[:8]}"
    
    exported_path = export_model(model_uri, output_dir, model_name)
    
    print(f"\n Model exported successfully!")
    print(f"   Location: {exported_path}")
    print(f"\n   To load this model:")
    print(f"   >>> import mlflow")
    print(f"   >>> model = mlflow.sklearn.load_model('{exported_path}')")


def compare_models_cmd(run_id1, run_id2, artifact_path="model"):
    """Compare two models."""
    model_uri1 = f"runs:/{run_id1}/{artifact_path}"
    model_uri2 = f"runs:/{run_id2}/{artifact_path}"
    
    print(f"\n🔍 Comparing models:")
    print(f"   Model 1: {run_id1}")
    print(f"   Model 2: {run_id2}\n")
    
    compare_models(model_uri1, model_uri2)


def model_size_cmd(run_id, artifact_path="model"):
    """Get model size."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    size = get_model_size(model_uri)
    
    print(f"\n📦 Model size: {size}")
    print(f"   Run ID: {run_id}")


def main():
    parser = argparse.ArgumentParser(
        description="MLflow Models Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python manage_models.py list-experiments
  
  # List runs in experiment
  python manage_models.py list-runs --experiment-id 0
  
  # Inspect model
  python manage_models.py inspect --run-id <run_id>
  
  # Export model
  python manage_models.py export --run-id <run_id> --output ../models --name my_model
  
  # Compare two models
  python manage_models.py compare --run-id1 <run_id1> --run-id2 <run_id2>
  
  # Get model size
  python manage_models.py size --run-id <run_id>
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List experiments
    subparsers.add_parser('list-experiments', help='List all experiments')
    
    # List runs
    list_runs_parser = subparsers.add_parser('list-runs', help='List runs in experiment')
    list_runs_parser.add_argument('--experiment-id', required=True, help='Experiment ID')
    
    # Inspect model
    inspect_parser = subparsers.add_parser('inspect', help='Inspect model structure')
    inspect_parser.add_argument('--run-id', required=True, help='Run ID')
    inspect_parser.add_argument('--artifact-path', default='model', help='Artifact path (default: model)')
    
    # Export model
    export_parser = subparsers.add_parser('export', help='Export model to directory')
    export_parser.add_argument('--run-id', required=True, help='Run ID')
    export_parser.add_argument('--output', required=True, help='Output directory')
    export_parser.add_argument('--name', help='Model name')
    export_parser.add_argument('--artifact-path', default='model', help='Artifact path (default: model)')
    
    # Compare models
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('--run-id1', required=True, help='First run ID')
    compare_parser.add_argument('--run-id2', required=True, help='Second run ID')
    compare_parser.add_argument('--artifact-path', default='model', help='Artifact path (default: model)')
    
    # Model size
    size_parser = subparsers.add_parser('size', help='Get model size')
    size_parser.add_argument('--run-id', required=True, help='Run ID')
    size_parser.add_argument('--artifact-path', default='model', help='Artifact path (default: model)')
    
    args = parser.parse_args()
    
    # Set tracking URI
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:./notebooks/mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    
    # Execute command
    if args.command == 'list-experiments':
        list_experiments()
    
    elif args.command == 'list-runs':
        list_runs(args.experiment_id)
    
    elif args.command == 'inspect':
        inspect_model(args.run_id, args.artifact_path)
    
    elif args.command == 'export':
        export_model_cmd(args.run_id, args.output, args.name, args.artifact_path)
    
    elif args.command == 'compare':
        compare_models_cmd(args.run_id1, args.run_id2, args.artifact_path)
    
    elif args.command == 'size':
        model_size_cmd(args.run_id, args.artifact_path)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

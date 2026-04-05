"""
MLflow Model inspection and export utilities.
"""
import os
import shutil
import yaml
import json
import mlflow
import mlflow.artifacts
from pathlib import Path


def inspect_model_structure(model_uri, verbose=True):
    """
    Download and inspect MLflow Model structure.
    
    Parameters:
    -----------
    model_uri : str
        MLflow model URI (e.g., 'runs:/<run_id>/model')
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    dict : Model structure information
    """
    # Download model artifacts
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    structure = {
        'local_path': local_path,
        'files': [],
        'mlmodel': None,
        'conda_env': None,
        'requirements': None,
        'input_example': None
    }
    
    # List all files
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, local_path)
            structure['files'].append(rel_path)
    
    # Read MLmodel file
    mlmodel_path = os.path.join(local_path, 'MLmodel')
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path, 'r') as f:
            structure['mlmodel'] = yaml.safe_load(f)
    
    # Read conda.yaml
    conda_path = os.path.join(local_path, 'conda.yaml')
    if os.path.exists(conda_path):
        with open(conda_path, 'r') as f:
            structure['conda_env'] = yaml.safe_load(f)
    
    # Read requirements.txt
    req_path = os.path.join(local_path, 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            structure['requirements'] = f.read().strip().split('\n')
    
    # Read input_example.json
    example_path = os.path.join(local_path, 'input_example.json')
    if os.path.exists(example_path):
        with open(example_path, 'r') as f:
            structure['input_example'] = json.load(f)
    
    if verbose:
        print("=" * 70)
        print(f"MLflow Model Structure: {model_uri}")
        print("=" * 70)
        print(f"\n📁 Local path: {local_path}\n")
        
        print("📄 Files in model:")
        for file in sorted(structure['files']):
            file_size = os.path.getsize(os.path.join(local_path, file))
            print(f"  - {file} ({file_size:,} bytes)")
        
        if structure['mlmodel']:
            print(f"\n🏷️  Model flavors: {list(structure['mlmodel'].get('flavors', {}).keys())}")
            print(f"⏰ Created: {structure['mlmodel'].get('utc_time_created', 'N/A')}")
        
        if structure['conda_env']:
            print(f"\n🐍 Python version: {structure['conda_env'].get('dependencies', [{}])[0]}")
        
        if structure['requirements']:
            print(f"\n📦 Requirements ({len(structure['requirements'])} packages):")
            for req in structure['requirements'][:5]:
                print(f"  - {req}")
            if len(structure['requirements']) > 5:
                print(f"  ... and {len(structure['requirements']) - 5} more")
        
        print("\n" + "=" * 70)
    
    return structure


def export_model(model_uri, output_dir, model_name=None):
    """
    Export MLflow Model to a specific directory.
    
    Parameters:
    -----------
    model_uri : str
        MLflow model URI
    output_dir : str
        Directory to export model to
    model_name : str, optional
        Name for the exported model directory
        
    Returns:
    --------
    str : Path to exported model
    """
    # Download model
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    # Determine output path
    if model_name is None:
        model_name = f"model_{Path(local_path).parent.name}"
    
    output_path = os.path.join(output_dir, model_name)
    
    # Copy model to output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    shutil.copytree(local_path, output_path)
    
    print(f" Model exported to: {output_path}")
    return output_path


def print_mlmodel_file(model_uri):
    """
    Print the contents of MLmodel file.
    
    Parameters:
    -----------
    model_uri : str
        MLflow model URI
    """
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    mlmodel_path = os.path.join(local_path, 'MLmodel')
    
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path, 'r') as f:
            content = f.read()
        
        print("=" * 70)
        print("MLmodel File Content")
        print("=" * 70)
        print(content)
        print("=" * 70)
    else:
        print(" MLmodel file not found")


def print_conda_yaml(model_uri):
    """
    Print the contents of conda.yaml file.
    
    Parameters:
    -----------
    model_uri : str
        MLflow model URI
    """
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    conda_path = os.path.join(local_path, 'conda.yaml')
    
    if os.path.exists(conda_path):
        with open(conda_path, 'r') as f:
            content = f.read()
        
        print("=" * 70)
        print("conda.yaml File Content")
        print("=" * 70)
        print(content)
        print("=" * 70)
    else:
        print(" conda.yaml file not found")


def compare_models(model_uri1, model_uri2):
    """
    Compare two MLflow Models.
    
    Parameters:
    -----------
    model_uri1, model_uri2 : str
        MLflow model URIs to compare
        
    Returns:
    --------
    dict : Comparison results
    """
    struct1 = inspect_model_structure(model_uri1, verbose=False)
    struct2 = inspect_model_structure(model_uri2, verbose=False)
    
    comparison = {
        'same_flavors': set(struct1['mlmodel'].get('flavors', {}).keys()) == 
                       set(struct2['mlmodel'].get('flavors', {}).keys()),
        'flavors_1': list(struct1['mlmodel'].get('flavors', {}).keys()),
        'flavors_2': list(struct2['mlmodel'].get('flavors', {}).keys()),
        'same_files': set(struct1['files']) == set(struct2['files']),
        'files_only_in_1': list(set(struct1['files']) - set(struct2['files'])),
        'files_only_in_2': list(set(struct2['files']) - set(struct1['files'])),
    }
    
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"\n🏷️  Model 1 flavors: {comparison['flavors_1']}")
    print(f"🏷️  Model 2 flavors: {comparison['flavors_2']}")
    print(f" Same flavors: {comparison['same_flavors']}")
    
    print(f"\n📄 Same files: {comparison['same_files']}")
    if comparison['files_only_in_1']:
        print(f"   Only in Model 1: {comparison['files_only_in_1']}")
    if comparison['files_only_in_2']:
        print(f"   Only in Model 2: {comparison['files_only_in_2']}")
    
    print("=" * 70)
    
    return comparison


def get_model_size(model_uri):
    """
    Get total size of MLflow Model.
    
    Parameters:
    -----------
    model_uri : str
        MLflow model URI
        
    Returns:
    --------
    int : Size in bytes
    """
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    total_size = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    # Convert to human-readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.2f} {unit}"
        total_size /= 1024.0
    
    return f"{total_size:.2f} TB"

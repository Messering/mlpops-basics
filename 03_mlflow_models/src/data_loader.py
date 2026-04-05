"""
Data loading utilities for MLflow Models demonstrations.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split


def load_sample_data(dataset_name='iris', test_size=0.2, random_state=42):
    """
    Load sample datasets for demonstrations.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset to load ('iris', 'wine', 'breast_cancer')
    test_size : float
        Proportion of dataset to include in test split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split dataset
    feature_names : list
        Names of features
    target_names : list
        Names of target classes
    """
    # Load dataset
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, 
        test_size=test_size, 
        random_state=random_state,
        stratify=data.target
    )
    
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names


def get_sample_input(X, n_samples=5):
    """
    Get sample input for model signature.
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_samples : int
        Number of samples to return
        
    Returns:
    --------
    pandas.DataFrame
        Sample input data
    """
    return pd.DataFrame(X[:n_samples])

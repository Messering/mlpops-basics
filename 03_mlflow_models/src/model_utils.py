"""
Model utilities for MLflow Models demonstrations.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def create_sklearn_model(model_type='random_forest', **kwargs):
    """
    Create a scikit-learn model.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('random_forest', 'logistic_regression', 'decision_tree')
    **kwargs : dict
        Model parameters
        
    Returns:
    --------
    model : sklearn estimator
        Initialized model
    """
    if model_type == 'random_forest':
        default_params = {'n_estimators': 100, 'random_state': 42, 'max_depth': 5}
        default_params.update(kwargs)
        return RandomForestClassifier(**default_params)
    
    elif model_type == 'logistic_regression':
        default_params = {'random_state': 42, 'max_iter': 1000}
        default_params.update(kwargs)
        return LogisticRegression(**default_params)
    
    elif model_type == 'decision_tree':
        default_params = {'random_state': 42, 'max_depth': 5}
        default_params.update(kwargs)
        return DecisionTreeClassifier(**default_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

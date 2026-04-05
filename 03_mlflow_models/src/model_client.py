"""
REST API client for MLflow model serving.

Provides a simple interface to interact with a locally served MLflow model.
"""
import requests
import json
import pandas as pd
import numpy as np


class MLflowModelClient:
    """Client for MLflow model REST API."""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip("/")

    def ping(self):
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/ping", timeout=2)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def predict(self, data):
        """Send a prediction request.

        Parameters
        ----------
        data : pd.DataFrame
            Input data as a DataFrame.

        Returns
        -------
        list
            Predictions from the model.
        """
        if isinstance(data, pd.DataFrame):
            payload = {"dataframe_split": data.to_dict(orient="split")}
        elif isinstance(data, np.ndarray):
            payload = {"dataframe_split": pd.DataFrame(data).to_dict(orient="split")}
        else:
            raise ValueError("data must be a pandas DataFrame or numpy array")

        response = requests.post(
            f"{self.base_url}/invocations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def predict_single(self, data, feature_names=None):
        """Send a single-sample prediction request.

        Parameters
        ----------
        data : pd.DataFrame or array-like
            Single sample input data.
        feature_names : list, optional
            Column names (used when data is array-like).

        Returns
        -------
        Prediction for the single sample.
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data.reshape(1, -1), columns=feature_names)
        else:
            df = pd.DataFrame([data], columns=feature_names)

        result = self.predict(df)
        if isinstance(result, dict) and "predictions" in result:
            return result["predictions"][0]
        if isinstance(result, list):
            return result[0]
        return result

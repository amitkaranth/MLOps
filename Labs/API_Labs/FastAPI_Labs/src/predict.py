import joblib
from pydantic import BaseModel
from typing import List

class WineBatchRequest(BaseModel):
    samples: List[List[float]]

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/wine_model.pkl")
    y_pred = model.predict(X)
    return y_pred

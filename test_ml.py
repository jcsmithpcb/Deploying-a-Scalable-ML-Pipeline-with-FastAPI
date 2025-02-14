import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
import pandas as pd

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test if train_model function returns a trained RandomForestClassifier model
    """
    # Your code here
    X_train = np.array([[0, 1, 2], [1, 0, 3], [2, 1, 0]])
    y_train = np.array([0, 1, 0])

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "train_model did not return a RandomForestClassifier instance"
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test is inference function returns the correct number of predictions
    """
    # Your code here
    X_test = np.array([[0, 1, 2], [1, 0, 3], [2, 1, 0]])
    y_test = np.array([0, 1, 0])

    model = train_model(X_test, y_test)
    preds = inference(model, X_test)
    assert len(preds) == len(X_test), "Inference function did not return the correct number of predictions"
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test if compute_model_metrics function returns precision, recall, and fbeta score within valid ranges
    """
    # Your code here
    y_true = np.array([0, 1, 0, 1])
    y_preds = np.array([0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert 0 <= precision <= 1, "Precision is out of range"
    assert 0 <= recall <= 1, "Recall is out of range"
    assert 0 <= fbeta <= 1, "Fbeta score is out of range"    
    pass

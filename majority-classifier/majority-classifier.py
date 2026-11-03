import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    values, first_idx, counts = np.unique(y_train, return_index=True, return_counts=True)
    idx = np.lexsort((first_idx, -counts))[0]
    majority_class = values[idx] 

    return np.ones(len(X_test)) * majority_class
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n, m = X.shape
    w = np.zeros(m)
    b = 0.0

    for step in range(steps):
        z = X.dot(w) + b
        p = _sigmoid(z) 
        
        # Compute gradients
        grad_w = X.T.dot(p - y) / len(y)
        grad_b = np.mean(p - y)

        # Update
        w -= lr * grad_w
        b -= lr * grad_b
        
        if step % 100 == 0:
            print(f'Step: {step} | w: {w} | b: {b}')

    return w, b
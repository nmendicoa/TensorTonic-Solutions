import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.array(A)
    n, m = A.shape
    A_T = np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            A_T[j, i] = A[i, j]
    return A_T

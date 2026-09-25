import numpy as np
from typing import List, Union, Any

def relu(x: Union[List[Any], int]) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    return np.array(np.maximum(0.0, x))
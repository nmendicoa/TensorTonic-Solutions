import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.array(x)
    mu = np.mean(x)
    std = np.std(x)

    centered = x - mu
    s = (np.sum(centered ** 2) / (x.size - 1))
    
    return {
        "variance": float(s), 
        "standard_deviation": float(np.sqrt(s))
    }
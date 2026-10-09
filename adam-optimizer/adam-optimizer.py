import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # 0: Prepare inputs
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    
    # 1: Update First Moment
    m_t = beta1 * m + (1 - beta1) * grad

    # 2: Update Second Moment
    v_t = beta2 * v + (1 - beta2) * grad ** 2

    # 3: Bias Correction
    m_t_hat = m_t / (1 - beta1 ** t) 
    v_t_hat = v_t / (1 - beta2 ** t)

    # 4: Parameter Update
    param = param - lr * m_t_hat / (np.sqrt(v_t_hat) + eps)
    
    return param, m_t, v_t
def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = x0
    
    for step in range(steps):
        grad_f = 2 * a * x + b
        x -= lr * grad_f
        if step % 10 == 0:
            print(f'Step: {step} | grad_f: {grad_f} | x: {x}')

    return x
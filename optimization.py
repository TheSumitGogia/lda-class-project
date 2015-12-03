import numpy as np

def newton(init, func, grad='approx', hess='approx', step='bls'):
    def bls(curr, change, alpha=0.25, beta=0.5):
        t, guess = 1, None
        while (guess is None or guess >= bound):
            guess = func(curr + t * change)
            bound = func(curr) + alpha * t * (grad(curr)).T.dot(change)
            t = beta * t
        return t

    dist_est = None
    current = init
    while (dist_est is None or dist_est > tolerance):
        gradient, hessian = grad(current), hess(current)
        change = -1 * np.linalg.inv(hessian).dot(gradient)
        step = bls(current, change) if step == 'bls' else step
        current += step * change
        dist_est = (gradient.T).dot(-1 * change)

    return current, func(current)

import numpy as np

def newton(init, func, grad='approx', hess='approx', step='bls'):
    '''
    def bls(curr, change, gradient, alpha=0.01, beta=0.9):
        t, guess = 1, None
        while (guess is None or guess >= bound):
            print 'caught'
            guess = func(curr + t * change)
            bound = func(curr) + alpha * t * (gradient.T.dot(change))
            t = beta * t
        return t
    '''

    dist_est = None
    current = init.copy()
    tolerance = 0.0001
    while (dist_est is None or dist_est > tolerance):
        print current
        gradient, hessian = grad(current), hess(current)
        change = -1 * np.linalg.inv(hessian).dot(gradient)
        step = bls(current, change, gradient) if step == 'bls' else step
        current += 1 * change
        dist_est = ((gradient.T).dot(-1 * change)) ** 0.5

    return current, func(current)

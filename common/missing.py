import numpy as np
from scipy.stats import norm

def get_missing_probabilities(mat, conf=0.95, tol=0.1):
    p, n = np.shape(mat)

    delta = np.mean(np.array(np.abs(mat) > 0.00000001, dtype=float), axis=1)
    stderr = np.sqrt((delta - delta * delta) / n) * norm.ppf(conf)
    idx = np.where(delta * tol >= stderr / 2)[0]

    return delta, stderr, idx


def missing_var(X, delta, make_positive=True):
    n = np.shape(X)[1]
    S = np.matmul(X, X.T) / n
    D = np.diag(np.diag(S))
    Delta = np.diag(delta ** (-1))
    Sigma = np.matmul(Delta, np.matmul(S - D, Delta)) + np.matmul(Delta, D)

    if make_positive:
        w, v = np.linalg.eigh(Sigma)
        w_new = np.maximum(w, 0)
        return np.matmul(np.matmul(v, np.diag(w_new)), v.T)
    else:
        return Sigma


def missing_covar(X, Y, delta_x, delta_y):
    assert(np.shape(X)[1] == np.shape(Y)[1])
    n = np.shape(X)[1]

    return np.matmul(np.matmul(np.diag(delta_x ** (-1)), X), np.matmul(Y.T, np.diag(delta_y ** (-1)))) / n

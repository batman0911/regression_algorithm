import numpy as np


def cost_func(w, X, y):
    eps = y - np.dot(X, w)
    return float(np.dot(eps.T, eps) / 2)


def gradient(w, X, y):
    return - np.dot(X.T, y) + np.dot(np.dot(X.T, X), w)


def hessian(X):
    return np.dot(X.T, X)


def cal_direction(H, g):
    return np.linalg.solve(H, g)


def update(w, t, p):
    return w + t * p


def predict(X, w):
    return np.dot(X, w)

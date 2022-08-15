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


class Tmp:
    def __init__(self, name):
        self.name = name

    def show_name(self):
        print(f'hello dai ca {self.name}')


class RegressionOpt:
    def __init__(self,
                solver='gd',
                tol=1e-4,
                max_iter=10000,
                step_size=1,
                check_after=1,
                w=None,
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None,
                lost_func=None,
                grad_func=None,
                bench_mode=False,
                terminate=False):
        self.tol = tol
        self.max_iter = max_iter
        self.step_size = step_size
        self.solver = solver
        self.grad_norm_list = []
        self.grad = None
        self.loss_func_list = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.loss_func = lost_func
        self.grad_func = grad_func
        self.w = w
        self.count = 0
        self.bench_mode = bench_mode
        self.terminate = terminate

    def fit_gd(self):
        t = -self.step_size / self.X_train.shape[0]
        while self.count < self.max_iter:
            self.count += 1
            grad = self.grad_func(self.w, self.X_train, self.y_train)
            self.w = update(self.w, t, grad)
            grn = np.linalg.norm(grad)
            if not self.bench_mode:
                self.grad_norm_list.append(grn)
                self.loss_func_list.append(self.loss_func(self.w, self.X_train, self.y_train))
            if not self.terminate:
                if grn < self.tol:
                    break

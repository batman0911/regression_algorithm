import numpy as np


def loss_func(w, X, y):
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


def back_tracking_step_size_gd(w, X, y, grad, t_init, alpha, beta):
    step_size = t_init
    count = 0
    old_cost = loss_func(w, X, y)
    grad_prd = alpha * np.dot(grad.T, grad)
    while loss_func(update(w, -step_size, grad), X, y) > \
            old_cost - step_size * grad_prd:
        step_size = beta * step_size
        count += 1
    return step_size, count


def back_tracking_step_size_acc(w, v, X, y, grad, t_init, beta):
    pass
    gv = loss_func(v, X, y)
    t = t_init
    next_w = update(v, -t, grad)
    d_w_v = next_w - v
    count = 0
    while loss_func(next_w, X, y) > gv + np.dot(grad.T, (w - v)) + 1 / (2 * t) * np.dot(d_w_v.T, d_w_v):
        count += 1
        t = beta * t
        next_w = update(v, -t, grad)
        d_w_v = next_w - v
    return t, count


class RegressionOpt:
    def __init__(self,
                 solver='gd',
                 backtracking=False,
                 tol=1e-4,
                 max_iter=10000,
                 step_size=1,
                 check_after=1,
                 alpha=1e-4,
                 beta=0.5,
                 w=None,
                 X_train=None,
                 y_train=None,
                 X_test=None,
                 y_test=None,
                 bench_mode=False,
                 terminate=True):
        self.solver = solver
        self.backtracking = backtracking
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
        self.w = w
        self.count = 0
        self.bench_mode = bench_mode
        self.terminate = terminate
        self.inner_count = 0
        self.alpha = alpha
        self.beta = beta

        self.setup()

    def setup(self):
        grn = np.linalg.norm(gradient(self.w, self.X_train, self.y_train))
        loss = loss_func(self.w, self.X_train, self.y_train)
        self.grad_norm_list.append(grn)
        self.loss_func_list.append(loss)

    def fit_gd(self):
        t_init = self.step_size / self.X_train.shape[0]
        t = t_init
        while self.count < self.max_iter:
            self.count += 1
            grad = gradient(self.w, self.X_train, self.y_train)
            if self.backtracking:
                t, inner_count = back_tracking_step_size_gd(self.w, self.X_train, self.y_train, grad,
                                                            t_init, self.alpha, self.beta)
                self.inner_count += inner_count
            self.w = update(self.w, -t, grad)
            grn = np.linalg.norm(grad)
            if not self.bench_mode:
                self.grad_norm_list.append(grn)
                self.loss_func_list.append(loss_func(self.w, self.X_train, self.y_train))
            if self.terminate and grn < self.tol:
                break
        return self.w

    def fit_newton(self):
        t = self.step_size
        while self.count < self.max_iter:
            self.count += 1
            grad = gradient(self.w, self.X_train, self.y_train)
            H = hessian(self.X_train)
            p = cal_direction(H, grad)
            self.w = update(self.w, -t, p)
            self.loss_func_list.append(loss_func(self.w, self.X_train, self.y_train))
            grn = np.linalg.norm(grad)
            self.grad_norm_list.append(grn)
            if self.terminate and grn < self.tol:
                break
        return self.w

    def fit_acc_gd(self):
        t_init = self.step_size / self.X_train.shape[0]
        t = t_init
        w1 = update(self.w, t, gradient(self.w, self.X_train, self.y_train))

        w = [self.w, w1]

        self.count += 1
        while self.count < self.max_iter:
            self.count += 1
            v = w[1] + (self.count - 2) / (self.count + 1) * (w[1] - w[0])
            w[0] = w[1]
            grad = gradient(v, self.X_train, self.y_train)
            if self.backtracking:
                t, inner_count = back_tracking_step_size_acc(self.w, v, self.X_train, self.y_train, grad,
                                                             t_init, self.beta)
                self.inner_count += inner_count
            w[1] = update(v, -t, grad)
            self.w = w[1]
            self.loss_func_list.append(loss_func(w[1], self.X_train, self.y_train))
            grn = np.linalg.norm(grad)
            self.grad_norm_list.append(grn)
            if self.terminate and grn < self.tol:
                break

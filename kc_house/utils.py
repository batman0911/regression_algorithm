import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    dat = pd.read_csv(file_name)
    N = dat.shape[0]

    yrs_old = np.zeros(N)
    yrs_reno = np.zeros(N)
    for i in range(N):
        year_sell = int(dat['date'][i][-4:])
        yrs_old[i] = year_sell - dat['yr_built'][i]
        if dat['yr_renovated'][i] == 0:
            yrs_reno[i] = 0
        else:
            yrs_reno[i] = year_sell - dat['yr_renovated'][i]

    y = np.array(dat['price']).reshape((N, 1))

    yrs_old = yrs_old.reshape((N, 1))
    yrs_reno = yrs_reno.reshape((N, 1))
    X = dat.to_numpy(dat.drop(columns=['id', 'date', 'price', 'yr_built',\
                                       'yr_renovated','zipcode', 'lat', 'long'], inplace=True)) # 'zipcode', 'lat', 'long'
    X = np.concatenate((X, yrs_old, yrs_reno), axis=1)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # Xy = np.concatenate((X,y),axis=1)
    # cols = list(dat.columns)
    # cols.extend(('years_built','years_renovated','price'))
    # dat_out = pd.DataFrame(Xy,columns=cols)
    # dat_out.to_csv('kc_house_data_cleaned.csv', index=False)

    return X, y

def loss_func(X, y, w):
    # loss = .5 * (np.linalg.norm(y-np.dot(X,w)))**2
    # return loss/X.shape[0]
    eps = y - np.dot(X, w)
    return float(np.dot(eps.T, eps) / 2)

def grad(X, y, w):
    # return np.dot(X.T, np.dot(X,w)-y)/X.shape[0]
    return np.dot(X.T, np.dot(X, w) - y)

def hess(X):
    return np.dot(X.T, X)/X.shape[0]

def predict(X,w):
    y_hat = np.dot(X,w)
    return y_hat

def GD(X, y, w_init, step_size, check_after, max_iter, tol=1e-4):
    w = w_init
    norm_grad = []
    loss = []
    step_size = step_size / X.shape[0]
    for i in range(max_iter):
        g = grad(X,y,w)
        w = w - step_size * g
        if i % check_after == 0:
            loss.append(loss_func(X, y, w))
            norm_grad.append(np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break
    return [w, i, norm_grad, loss]

def bktrack_step_GD(X,y,w,t_init,grad,alpha, beta):
    t = t_init
    while loss_func(X,y,w - t * grad) > loss_func(X,y,w) - alpha * t * (np.linalg.norm(grad)**2):
        t = beta * t
    return t

def bktrack_step_Newton(X,y,w,t_init,grad,v,alpha, beta):
    t = t_init
    while loss_func(X,y,w + t * v) > loss_func(X,y,w) + alpha * t * np.dot(grad.T,v):
        t = beta * t
    return t

def BGD(X, y, w_init, step_size, check_after, max_iter, tol=1e-4):
    w = w_init
    norm_grad = []
    loss = []
    alpha = .5
    beta = .5
    for i in range(max_iter):
        g = grad(X,y,w)
        t = bktrack_step_GD(X, y, w, step_size, g, alpha, beta)
        w = w - t * g

        if i % check_after == 0:
            # loss.append(loss_func(X, y, w))
            # norm_grad.append(np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break
    return [w, i, norm_grad, loss]

def Newton(X, y, w_init, check_after, max_iter, tol=1e-4):
    w = w_init
    norm_grad = []
    loss = []
    for i in range(max_iter):
        g = grad(X,y,w)
        h = hess(X)
        w = w - np.dot(np.linalg.inv(h), g)

        if i % check_after == 0:
            # loss.append(loss_func(X, y, w))
            # norm_grad.append(np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break
    return [w, i, norm_grad, loss]

def BNewton(X, y, w_init, step_size, check_after, max_iter, tol=1e-4):
    w = w_init
    norm_grad = []
    loss = []
    alpha = .5
    beta = .5
    for i in range(max_iter):
        g = grad(X,y,w)
        h = hess(X)
        v = -np.dot(np.linalg.inv(h), g)
        t = bktrack_step_Newton(X, y, w, step_size, g, v, alpha, beta)
        w = w + t * v

        if i % check_after == 0:
            # loss.append(loss_func(X, y, w))
            # norm_grad.append(np.linalg.norm(g))
            if np.linalg.norm(g) < tol:
                break
    return [w, i, norm_grad, loss]

def plot_data(gradient, loss):
    # plt.plot(range(len(loss)),loss)
    plt.plot(range(50), loss[1:51])
    plt.title(f'Loss function ', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()
    # plt.plot(range(len(gradient)),gradient)
    plt.plot(range(50), gradient[1:51])
    plt.title(f'Gradient norm - ', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import regression as reg
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('../data/input/kc_house_data_cleaned.csv')

    X = df[df.columns[2:3]].to_numpy()
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)
    y = df['price'].to_numpy()
    # y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y / 1e6
    y = y.reshape((X.shape[0], 1))

    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    w_init = np.repeat(1, X_train.shape[1]).reshape((X_train.shape[1], 1))

    # %%
    lm = reg.RegressionOpt(
        max_iter=100,
        w=w_init,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    lm.fit_newton()

    loss_func_list = lm.loss_func_list
    grad_norm_list = lm.grad_norm_list

    plt.plot(range(len(loss_func_list)), loss_func_list)
    plt.title(f'Loss function', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()

    plt.plot(range(len(grad_norm_list)), grad_norm_list)
    plt.title(f'Gradient norm', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()

    print(f'count: {lm.count}, gradient norm: {grad_norm_list[-1]}')
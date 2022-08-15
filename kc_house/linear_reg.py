from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import *
import time

X, y = load_data('kc_house_data.csv')
# print(X[:5,:])
# print(y[:5])
one = np.ones((X.shape[0],1))
X = np.concatenate((one,X), axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=11)

# t1 = time.time()
# model = LinearRegression()
# model.fit(X_train, y_train)
# t2 = time.time()
# t_sk = t2 - t1
# y_hat = model.predict(X_test)
# print("Linear Regression of sklearn after {:.3f}s".format(t_sk))
# print("MSE : ",mean_squared_error(y_test, y_hat))
# print("Score : ",r2_score(y_test, y_hat))
# print("w: ",model.coef_)

# config
# w_init = np.ones((X_train.shape[1],1))
w_init = np.zeros((X_train.shape[1],1))
step_size = .1
check_after = 1
max_iter = 100000
tolerance = 1e-4

# t1 = time.time()
# w, i, g, l = GD(X,y,w_init, step_size, check_after, max_iter, tolerance)
# t2 = time.time()
# y_hat = predict(X_test,w)
# print("GD:")
# print("GD complete in ", t2-t1, " s, after ", i, " steps")
# print("MSE : ",mean_squared_error(y_test, y_hat))
# print("Score : ",r2_score(y_test, y_hat))
# plot_data(g,l)

# t1 = time.time()
# w, i, g, l = BGD(X,y,w_init, step_size, check_after, max_iter, tolerance)
# t2 = time.time()
# y_hat = predict(X_test,w)
# print("BGD:")
# print("BGD complete in ", t2-t1, " s, after ", i, " steps")
# print("MSE : ",mean_squared_error(y_test, y_hat))
# print("Score : ",r2_score(y_test, y_hat))
# plot_data(g,l)

# t1 = time.time()
# w, i, g, l = Newton(X,y,w_init, check_after, max_iter, tolerance)
# t2 = time.time()
# y_hat = predict(X_test,w)
# print("Newton:")
# print("Newton complete in ", t2-t1, " s, after ", i, " steps")
# print("MSE : ",mean_squared_error(y_test, y_hat))
# print("Score : ",r2_score(y_test, y_hat))
# plot_data(g,l)

# t1 = time.time()
# w, i, g, l = BNewton(X,y,w_init, step_size, check_after, max_iter, tolerance)
# t2 = time.time()
# y_hat = predict(X_test,w)
# print("Backtracking Newton:")
# print("Backtracking Newton complete in ", t2-t1, " s, after ", i, " steps")
# print("MSE : ",mean_squared_error(y_test, y_hat))
# print("Score : ",r2_score(y_test, y_hat))
# # plot_data(g,l)





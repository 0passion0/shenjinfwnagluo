import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def cost0(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # 将参数转换为矩阵形式
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
  # 前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # 计算损失
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    return J
def cost(params, input_size, hidden_size, num_labels, X, y,learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # 将参数转换为矩阵形式
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # 计算损失
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # 将参数转换为矩阵形式
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # 前向传播

    # 初始化
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # 计算损失
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # 损失函数正则化项
    j=cost(params, input_size, hidden_size, num_labels, X, y_onehot,learning_rate)
    # 反向传播
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    # 梯度矩阵分解为数组
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return j, grad

def backpropzengze(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # 将参数转换为矩阵形式
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # 前向传播
    # 初始化
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # 计算损失
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # 损失函数正则化项
    j=cost(params, input_size, hidden_size, num_labels, X, y_onehot,learning_rate)
    # 反向传播
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1 = delta1 / m
    delta2 = delta2 / m
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    # 梯度矩阵分解为数组
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return j, grad
if __name__ == '__main__':
    data = loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']

    print(X.shape, y.shape)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)
    print(y_onehot.shape)
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1


    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    # 最小化目标函数
    fmin = minimize(fun=backpropzengze, x0=params,
                    args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True)
    print(fmin)

    X = np.matrix(X)
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    print(y_pred)

    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
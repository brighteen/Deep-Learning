import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def sigmoid_diff(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def mean_squared_error(y, t):
    return (y-t)**2

x = 2
w = np.random.randn(1)
b = 0
t = 1
lr = 0.01

for i in range(1, 1001):
    if i % 100 == 0:
        print(f'[{i}번째] w: {w}, b: {b}')
    # print(f'[{i}번째] w: {w}, b: {b}')

    # forward
    y = x * w + b 
    z = sigmoid(y)
    loss = mean_squared_error(z, t)
    if i % 100 == 0:
        print(f'[{i}번째] z : {z}, loss: {loss}\n')
    # print(f'[{i}번째] y : {y}, z : {z}, loss: {loss}\n')

    dLdz = 2 * (z - t) # dL/dz = 2 * (z - t)
    dzdy = sigmoid_diff(y) # sigmoid(y) * (1 - sigmoid(y)) = sigmoid(y) * sigmoid_diff(y)

    dydw = x
    dydb = 1

    dLdw = dydw * dzdy * dLdz
    dLdb = dydb * dzdy * dLdz

    # update
    w = w - lr * dLdw
    b = b - lr * dLdb
import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def mean_squared_error(y, t):
    return np.sum((y-t)**2)

x = 2
w = np.random.randn(1)
b = 0
t = 1 # 정답
lr = 0.01

y = x * w + b # forward
z = sigmoid(y)
loss = mean_squared_error(z, t)
print(f'y, z, loss: {y}, {z}, {loss}')

dw = x * sigmoid_grad(y) * (2*(z - t)) # dL/dy * dy/dw
db = 1 * sigmoid_grad(y) * (2*(z - t))
print(f'\ndiff_loss_w: {dw}')
print(f'db: {db}')

w = w - lr * dw
b = b - lr * db

loss = mean_squared_error(z, t)
print(f'\n첫 loss: {loss}')

for i in range(1000):
    if i % 100 == 0:
        print('epoch: {}, loss: {}'.format(i, loss))
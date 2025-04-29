import numpy as np
np.random.seed(0)
x = 2
t = 1
lr = 0.01

w = np.random.randn(1)
b = np.zeros(1)

# print(f'Initial w: {w}, b: {b}')
# 딕셔너리로 w, b 표현 - Neural Network에서 학습시켜야 하는 애들만 딕셔너리로 표현현

param = {'W': w, 'b': b}
print(f'Initial w: {param["W"]}, b: {param["b"]}')

for i in range(3):
    # forward
    # y = w*x+b
    y = param['W'] * x + param['b']
    loss = (y-t)**2
    print(f'[{i+1}번째] y : {y}, loss: {loss}')

    # backward
    dL_dy = 2*(y-t)
    
    dy_dw = x
    dy_db = 1

    dw = dL_dy * dy_dw
    db = dL_dy * dy_db

    # update
    # w = w - lr*dw
    # b = b - lr*db

    param['W'] = param['W'] - lr*dw
    param['b'] = param['b'] - lr*db
    
    # print(f'[{i+1}번째] w: {w}, b: {b}')
    print(f'[{i+1}번째] w: {param["W"]}, b: {param["b"]}')

# print(f'\nFinal w: {w}, b: {b}')
print(f'\nFinal w: {param["W"]}, b: {param["b"]}')
print(f'Final y: {y}, Final loss: {loss}')
import numpy as np

x = 2
w = np.random.randn(1)
b = 0
t = 1 # 정답
lr = 0.01

for i in range(1000):
    # forward
    y = x * w + b
    L = (y - t) ** 2 # MSE 손실함수

    if i % 100 == 0:
        print(f"{i}, y 값 : {y}, L 값: {L}")
        print(f"w: {w}, b: {b}")

    # backpropagation
    dL_dy = 2 * (y - t) # dL/dy = 2 * (y - t)
    dy_dw = x
    dy_db = 1

    dw = dL_dy * dy_dw # dL/dw = dL/dy * dy/dw
    db = dL_dy * dy_db # dL/db = dL/dy * dy/db

    # update
    w = w - lr * dw
    b = b - lr * db

    # if i % 100 == 0:
    #     # print('epoch: {}, loss: {}'.format(i, L))
    #     print(f'{i}번째 반복: y: {y}, L: {L}') # y, L: 0.0

y = w * x + b
L = (y - t) ** 2
print(f'\n마지막 업데이트 후 y: {y}, L: {L}') # y, L: 0.0
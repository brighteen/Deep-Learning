'''
미분 -> 순간변화율
변화율 : y = f(x)일 때, x가 a에서 b로 변화할 때 y의 변화량을 x의 변화량으로 나눈 것
dy/dx = (f(b) - f(a)) / (b - a)
극한 : b-a가 0에 가까워질 때의 변화율
미분계수 : dy/dx = lim (b->a) (f(b) - f(a)) / (b - a)

y = x^2의 미분계수
lim(x -> 0) 일 때
dy/dx = (f(x + h) - f(x)) / h
= (f(x + h) - f(x)) / (x + h - x)

'''

'''x0 = 3
print(f'x0: {x0}')
y = x0**2
print(f'y: x^2: {y}') # y: 9

dy = 2 * x0 # dy/dx = 2x
print(f'dy/dx: {dy}') # dy: 6

print('---' * 20)

import numpy as np
y = 0
z = 1 / (1 + np.exp(-y)) # sigmoid 함수
print(f'y: {y}') # y: 0
print(f'z: {z}') # z: 0.5

dy = z * (1 - z) # sigmoid 함수의 미분
print(f'dy: {dy}') # dy: 0.25

print('---' * 20)'''

x = 2
w = 1
b = 0
t = 1
lr = 0.01

for i in range(1, 4):
    print(f'[{i}번째] w: {w}, b: {b}') # w: 0.98, b: -0.01
    y = x*w + b
    loss = (y - t) ** 2
    print(f'[{i}번째] y : {y}, loss: {loss}')

    dydw = x
    dydb = 1

    w = w - lr * dydw
    b = b - lr * dydb

# dx = 1 * w # dy/dx = 1
# dw = x * 1 # dy/dw = x
# db = 1 * 1 # dy/db = 1
# print(f'dy/dx: {dx}') # dy/dx: 1
# print(f'dy/dw: {dw}') # dy/dw: 1
# print(f'dy/db: {db}') # dy/db: 1
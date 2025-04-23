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

x0 = 3
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

print('---' * 20)

x = 2
np.random.seed(1) # 난수 생성기 초기화
w = np.random.randn(1) # 0~1 사이의 난수 생성
b = 0

y = x*w + b
print(f'y: {y}')

dx = 1 * w # dy/dx = 1
dw = x * 1 # dy/dw = x
db = 1 * 1 # dy/db = 1
print(f'dy/dx: {dx}') # dy/dx: 1
print(f'dy/dw: {dw}') # dy/dw: 1
print(f'dy/db: {db}') # dy/db: 1
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

x = 2
w = 1
b = 0
t = 1
lr = 0.01

for i in range(1, 4):
    print(f'[{i}번째] w: {w}, b: {b}')

    # forward
    y = x*w + b
    loss = (y - t) ** 2
    print(f'[{i}번째] y : {y}, loss: {loss}\n')
    # backward
    dLdy = 2 * (y - t) # dL/dy = 2 * (y - t)
    dydw = x
    dydb = 1

    dLdw = dLdy * dydw # dL/dw = dL/dy * dy/dw
    dLdb = dLdy * dydb # dL/db = dL/dy * dy/db

    # update
    w = w - lr * dLdw
    b = b - lr * dLdb
'''
[1번째] w: 1, b: 0
[1번째] y : 2, loss: 1

[2번째] w: 0.96, b: -0.02
[2번째] y : 1.9, loss: 0.8099999999999998

[3번째] w: 0.9239999999999999, b: -0.038
[3번째] y : 1.8099999999999998, loss: 0.6560999999999997
'''
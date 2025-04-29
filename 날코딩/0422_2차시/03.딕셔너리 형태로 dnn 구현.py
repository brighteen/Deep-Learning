x = 2
w = 1
b = 0
t = 1
lr = 0.01
epochs = 3

A = {'W': w, 'b': b}

for i in range(1, epochs + 1):
    print(f'[{i}번째] w: {A["W"]}, b: {A["b"]}')

    # forward
    y = x * A['W'] + A['b']
    loss = (y - t) ** 2
    print(f'[{i}번째] y : {y}, loss: {loss}\n')
    
    # backward
    dLdy = 2 * (y - t)
    dydw = x
    dydb = 1

    dLdw = dLdy * dydw
    dLdb = dLdy * dydb

    # update
    A['W'] = A['W'] - lr * dLdw
    A['b'] = A['b'] - lr * dLdb

'''
[1번째] w: 1, b: 0
[1번째] y : 2, loss: 1

[2번째] w: 0.96, b: -0.02
[2번째] y : 1.9, loss: 0.8099999999999998

[3번째] w: 0.9239999999999999, b: -0.038
[3번째] y : 1.8099999999999998, loss: 0.6560999999999997
'''
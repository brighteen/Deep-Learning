import numpy as np
np.random.seed(1)

def simple_dnn(x, t, lr, epochs):
    w = np.random.randn(1)
    b = 0
    for i in range(1, epochs + 1):
        print(f'[{i}번째] w: {w}, b: {b}')

        # forward
        y = x * w + b
        loss = (y - t) ** 2
        print(f'[{i}번째] y : {y}, loss: {loss}\n')

        # backward
        dLdy = 2 * (y - t)
        dydw = x
        dydb = 1

        dLdw = dLdy * dydw
        dLdb = dLdy * dydb

        # update
        w = w - lr * dLdw
        b = b - lr * dLdb

    return w, b

if __name__ == "__main__":
    x = 2
    t = 1
    lr = 0.01
    epochs = 3

    final_w, final_b = simple_dnn(x, t, lr, epochs)
    print(f"최종 결과: w = {final_w}, b = {final_b}")
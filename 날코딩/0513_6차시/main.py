import numpy as np

from LinearModel import LinearModel
from MeanSquaredError import MeanSquardError

if __name__ == "__main__": # 다른 파일에서 main.py를 가져올 때 이 __main__안에 코드는 무시
    np.random.seed(0)

    x = 2
    t = 1

    # 두번의 선형변환, 손실함수 객체 선언
    model1 = LinearModel()
    model2 = LinearModel()
    mse = MeanSquardError()

    for i in range(0,3):
        print(f'\n===={i+1}번째 반복====')
        print(f"[model1] w1: {model1.w}, b1: {model1.b}")
        print(f"[model2] w2: {model2.w}, b2: {model2.b}")

        # forward
        out = model1.forward(x) # out = w1*x+b1
        print(f"y1: {out}")
        out = model2.forward(out) # out = w2*y1+b2
        print(f"y2: {out}")
        out = mse.forward(out, t) # out = (y2-t)^2
        print(f"loss: {out}")

        # backward
        dout = 1
        dout = mse.backward(dout=dout) # dout = dL/dy2
        print(f"dy2]: {dout}")
        dw2, db2, dout = model2.backward(dout) # dout = dL/dy1
        print(f"[dL/dw2]: {dw2}, [dL/db2]: {db2}, [dL/dy1]: {dout}")
        dw1, db1, dout = model1.backward(dout) # dout = dL/dx
        print(f"[dL/dw1]: {dw1}, [dL/db1]: {db1}, [dL/dx]: {dout}")

        # update, gradient descent(하강)
        lr = 0.01
        model2.w -= np.dot(lr, dw2)
        model2.b -= np.dot(lr, db2)
        model1.w -= np.dot(lr, dw1)
        model1.b -= np.dot(lr, db1)
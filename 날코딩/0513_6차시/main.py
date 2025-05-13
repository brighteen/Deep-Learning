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

    print(f"[model1] w1: {model1.w}, b1: {model1.b}")
    print(f"[model2] w2: {model2.w}, b2: {model2.b}")

    # forward
    out = model1.forward(x)
    print(f"y1[w1 * x + b1]: {out}")
    out = model2.forward(out)
    print(f"y2[w2 * y1 + b2]: {out}")
    out = mse.forward(out, t)
    print(f"loss[(y2 - t)^2]: {out}")

    # backward
    dout = 1
    dout = mse.backward(dout=dout) # dL/dy2
    print(f"[dL/dy2]: {dout}")
    dw2, db2, dx2 = model2.backward(dout) # dx2 = dL/dy1
    print(f"[dL/dw2]: {dw2}, [dL/db2]: {db2}, [dL/dy1]: {dx2}")
    dw1, db1, dx1 = model1.backward(dx2)
    print(f"[dL/dw1]: {dw1}, [db1]: {db1}, [dx1]: {dx1}")
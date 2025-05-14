import numpy as np
from common.LinearModel import LinearModel
from common.Sigmoid import Sigmoid
from common.MeanSquaredError import MeanSquaredError

if __name__ == "__main__":
    np.random.seed(0)

    x = 2
    t = 1
    lr = 0.1
    model1 = LinearModel()
    model2 = Sigmoid()
    mse = MeanSquaredError()

    for i in range(1, 4):
        # forward
        print(f"=== {i} ===")
        out = model1.forward(x)
        # print(f"y= (wx + b): {out}")
        out = model2.forward(out)
        print(f"예측값 z= 1/1+e^-y): {out}")
        out = mse.forward(out, t)
        print(f"손실값 loss= (z-t)^2: {out}")

        # backward
        dout=1
        dout = mse.backward(dout)
        # print(f"dL/dz: {dout}")
        dout = model2.backward(dout)
        # print(f"dL/dy: {dout}")
        dw, db, dx = model1.backward(dout)
        # print(f"dL/dw: {dw}, dL/db: {db}, dL/dx: {dx}")

        model1.w -= np.dot(lr, dw)
        model1.b -= np.dot(lr, db)
from common.layers import Affine
from common.layers import MeanSquaredError
import numpy as np

np.random.seed(42)

z = np.array([1,1])
z = np.reshape(z, (1,-1))
print(f"[z] : {z}, shape: {np.shape(z)}") # 1행 2열

t = np.zeros(1)
print(f"[t] : {t}, shape: {np.shape(t)}") # 1행 1열

lr = 0.01
W = np.random.rand(2, 1)
b = np.zeros(1)
print(f"[W] : {W}, shape: {np.shape(W)}") # 2행 1열
print(f"[b] : {b}, shape: {np.shape(b)}") # 1행 1열

affine = Affine(W, b)
mse = MeanSquaredError()

for i in range(1, 4):
    print(f"\n=== {i} ===")
    # forward
    out = affine.forward(z)
    print(f"예측값 y= wx + b: {np.round(out,3)}, y.shape: {np.shape(out)}") # 1행 1열
    out = mse.forward(out, t)
    print(f"손실값 loss= (y-t)^2: {np.round(out,3)}, loss.shape: {np.shape(out)}") # 1행 1열

    # backward
    dout = np.ones(1)
    # print(f"dL/dy: {dout}")
    dout = mse.backward(dout)
    # print(f"dL/dy: {dout}")
    dout = affine.backward(dout)
    # print(f"dL/dx: {dout}")

    # update
    affine.W = affine.W - lr * affine.dW
    affine.b = affine.b - lr * affine.db

    # print(f"[W] : {affine.W}, shape: {np.shape(affine.W)}") # 2행 1열
    # print(f"[b] : {affine.b}, shape: {np.shape(affine.b)}") # 1행 1열
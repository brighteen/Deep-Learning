import numpy as np
from common.layers import Affine, MeanSquaredError, Sigmoid, Relu

np.random.seed(0)
z = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])

print(f"입력 z: \n{z}, shape: {np.shape(z)}\n")

t = np.array([[0],
              [1],
              [1],
              [0]])
print(f"정답답 t: \n{t}, shape: {np.shape(t)}\n")

W = np.random.randn(2,2)
b = np.zeros(2,)


affine1 = Affine(W,b)
sigmoid = Sigmoid()
# mse = MeanSquaredError()
print(f"가중치 W: \n{affine1.W}, shape: {np.shape(affine1.W)}, \n편향 b: {affine1.b}, shape: {np.shape(affine1.b)}\n")

# forward
y = affine1.forward(z)
print(f"선형변환 y: \n{y}, shape: {np.shape(y)}\n")

out = sigmoid.forward(y)
print(f"비선형 변환 z: \n{out}, shape: {np.shape(out)}\n")

# out = mse.forward(out, t)
# print(f"손실값 L: \n{out}, shape: {np.shape(out)}\n")

# 역전파
dout = np.ones((4,2)) # dL/dL
print(f"dL/dL: \n{dout}, shape: {np.shape(dout)}\n")

# dout = mse.backward(dout)
# print(f"dL/dz: {out}")

dout = sigmoid.backward(dout)
print(f"dL/dy: \n{dout}, shape: {np.shape(dout)}\n")

## 렐루적용
relu = Relu()
print("\n[ReLU]")
print(f"[ReLU] 선형변환 y: \n{y}, shape: {np.shape(y)}\n")
out = relu.forward(y)
print(f"[ReLU] 비선형변환 z: \n{out}, shape: {np.shape(out)}\n")

dout = np.ones((4,2)) # dL/dL
dout = relu.backward(dout)
print(f"[ReLU] dL/dy: \n{dout}, shape: {np.shape(dout)}\n")
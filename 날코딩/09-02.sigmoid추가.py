from 날코딩.common.layers import Sigmoid

import numpy as np

'''
sigmoid
    - 입력값을 0과 1 사이로 변환하는 비선형 함수
    - 주로 신경망의 활성화 함수(activation function)로 사용됨
    - 수식: sigmoid(x) = 1 / (1 + exp(-x))

data -> 선형 변환 1(node 2) -> 비선형 변환(sigmoid) -> 선형 변환 2(node 1) -> 손실 함수

'''

# np.random.seed(0)
# features = 5

# sigmoid = Sigmoid()
# # z = np.array([0,0,0,0])
# # z = np.random.randint(0, 5, 20)
# z = np.zeros(features,)
# # print(f"z: {z}, shape: {z.shape}")
# z = np.reshape(z, (z.shape[0], -1))
# print(f"z: \n{z},\nshape: {z.shape}")

# out = sigmoid.forward(z)
# print(f"\nsigmoid.forward(x): \n{out},\nshape: {out.shape}")

# # dout = np.array([1,1,1,1])
# # dout = np.reshape(dout, (1,-1))
# dout = np.ones(features,)
# dout = np.reshape(dout, (dout.shape[0], -1))
# print(f"\ndout: \n{dout},\nshape: {dout.shape}")
# dout = sigmoid.backward(dout)
# print(f"\nsigmoid.backward(dout): \n{dout}, \nshape: {dout.shape}")


from 날코딩.common.my_layers import Affine, MeanSquaredError

z = np.array([0,0])
# print(f"z: {z}, shape: {z.shape}")
z = np.reshape(z, (1, 2))
print(f"z: {z}, shape: {z.shape}")

t = np.zeros(1)

W1 = np.random.randn(2,2)
b1 = np.zeros(2,)
affine1 = Affine(W1, b1)
print(f"w1: {affine1.W}, shape: {affine1.W.shape}")
print(f"b1: {affine1.b}, shape: {affine1.b.shape}")

W2 = np.random.randn(2,1)
b2 = np.zeros(1,)
affine2 = Affine(W2, b2)
print(f"w2: {affine2.W}, shape: {affine2.W.shape}")
print(f"b2: {affine2.b}, shape: {affine2.b.shape}")

sigmoid = Sigmoid()
mse = MeanSquaredError()

out = affine1.forward(z)
print(f"y^1: \n{out}, shape: {out.shape}")
out = sigmoid.forward(out)
print(f"z^1: \n{out}, shape: {out.shape}")
out = affine2.forward(out)
print(f"y^2: \n{out}, shape: {out.shape}")
out = mse.forward(out, t)
print(f"loss: {out}, shape: {out.shape}")

dout = np.ones(1, )
dout = mse.backward(dout)
print(f"dL/dL: \n{dout}, shape: {dout.shape}")
dout = affine2.backward(dout)
print(f"dL/dy^2: \n{dout}, shape: {dout.shape}")
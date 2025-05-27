import numpy as np

from common.layers import Affine, MeanSquaredError

'''
data(2,1) -> 선형 변환 1(node 3) -> 선형 변환 2(node 2) -> 선형 변환(node 1) -> 손실함수
z = [1,1]
t = 0
'''

np.random.seed(0)

z = np.array([1,1])
print(f"z: {z}, shape: {z.shape}")
z = np.reshape(z, (1,-1))
print(f"z: {z}, reshape: {z.shape}")
t = np.zeros(1,)
print(f"t: {t}, shape: {t.shape}")

lr = 0.01

W1 = np.random.randn(2,3)
b1 = np.zeros(3,)
affine1 = Affine(W1, b1)
print(f"W1: {affine1.W}, shape: {np.shape(affine1.W)}")
print(f"b1: {affine1.b}, shape: {np.shape(affine1.b)}")

W2 = np.random.randn(3,2)
b2 = np.zeros(2,)
affine2 = Affine(W2, b2)
print(f"W2: {affine2.W}, shape: {np.shape(affine2.W)}")
print(f"b2: {affine2.b}, shape: {np.shape(affine2.b)}")

W3 = np.random.randn(2,1)
b3 = np.zeros(1,)  
affine3 = Affine(W3, b3)
print(f"W3: {affine3.W}, shape: {np.shape(affine3.W)}")
print(f"b3: {affine3.b}, shape: {np.shape(affine3.b)}")

mse = MeanSquaredError()

for i in range(100):
    # print("\n전파")
    out = affine1.forward(z)
    # print(f"y^1: {out}, shape: {np.shape(out)}")
    out = affine2.forward(out)
    # print(f"y^2: {out}, shape: {np.shape(out)}")
    out = affine3.forward(out)
    # print(f"y^3: {out}, shape: {np.shape(out)}")
    loss = mse.forward(out, t)
    # print(f"loss: {loss}, shape: {np.shape(loss)}")
    print(f"[{i}번째] 예측값: y^3: {out}, 손실값: {loss}")

    # print("\n역전파파")
    dout = np.ones(1,)
    dout = mse.backward(dout)
    # print(f"dL/dy^3: {dout}, shape: {np.shape(dout)}")
    dout = affine3.backward(dout)
    # print(f"dL/dy^2: {dout}, shape: {np.shape(dout)}")
    dout = affine2.backward(dout)
    # print(f"dL/dy^1: {dout}, shape: {np.shape(dout)}")
    dout = affine1.backward(dout)
    # print(f"dL/dx: {dout}, shape: {np.shape(dout)}")

    # print("\n가중치, 편향 업데이트")
    # print(f"affine1.dW: {affine1.dW}, shape: {np.shape(affine1.dW)}")
    # print(f"affine1.db: {affine1.db}, shape: {np.shape(affine1.db)}")
    # print(f"affine2.dW: {affine2.dW}, shape: {np.shape(affine2.dW)}")
    # print(f"affine2.db: {affine2.db}, shape: {np.shape(affine2.db)}")
    # print(f"affine3.dW: {affine3.dW}, shape: {np.shape(affine3.dW)}")
    # print(f"affine3.db: {affine3.db}, shape: {np.shape(affine3.db)}")

    affine3.W = affine3.W - lr * affine3.dW
    affine3.b = affine3.b - lr * affine3.db
    affine2.W = affine2.W - lr * affine2.dW
    affine2.b = affine2.b - lr * affine2.db
    affine1.W = affine1.W - lr * affine1.dW
    affine1.b = affine1.b - lr * affine1.db
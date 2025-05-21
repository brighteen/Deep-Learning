import numpy as np

from common.layers import Affine, MeanSquaredError
'''
data -> 선형 변환(node 1) -> 손실함수
z = [1,1]
t = 0
'''

np.random.seed(0)

z = np.array([1,1])
print(f"z: {z}, z.shape: {np.shape(z)}")
z = np.reshape(z, (1, -1))
print(f"z: {z}, z.shape: {np.shape(z)}") # z: [[1 1]], z.shape: (1, 2)

t = np.array([0])
print(f"t: {t}, t.shape: {np.shape(t)}") # t: [0], t.shape: (1,)

W = np.random.randn(2,1) # (2,1) 행렬로 변환
b = np.zeros(1)
# print(f"W: {W}, W.shape: {np.shape(W)}")
# print(f"b: {b}, b.shape: {np.shape(b)}")

affine = Affine(W, b)
mse = MeanSquaredError()

print(f"W: {affine.W}, W.shape: {np.shape(affine.W)}") # W: [[1.76405235], [0.40015721]], W.shape: (2, 1)
print(f"b: {affine.b}, b.shape: {np.shape(affine.b)}") # b: [0.], b.shape: (1,)

# 순전파
out = affine.forward(z) # 예측값 y
print(f"out: {out}, out.shape: {np.shape(out)}") # out: [[2.16420955]], out.shape: (1, 1)

out = mse.forward(out, t) # 손실값 Loss
print(f"out: {out}, out.shape: {np.shape(out)}") # out: 2.341901497537206, out.shape: ()

# 역전파
dout = np.ones(1) # dL/dL
print(f"dout: {dout}, dout.shape: {np.shape(dout)}") # dout: [1.], dout.shape: (1,)

dout = mse.backward(dout) # dL/dy
print(f"dout: {dout}, dout.shape: {np.shape(dout)}") # dout: [[2.16420955]], dout.shape: (1, 1)

dout = affine.backward(dout) # dL/dx
print(f"dout: {dout}, dout.shape: {np.shape(dout)}") # dout: [[3.81777894 0.86602405]], dout.shape: (1, 2)

# dW, db 확인
print(f"affine.dW: {affine.dW}, affine.dW.shape: {np.shape(affine.dW)}") # affine.dW: [[2.16420955], [2.16420955]], affine.dW.shape: (2, 1)
print(f"affine.db: {affine.db}, affine.db.shape: {np.shape(affine.db)}") # affine.db: [2.16420955], affine.db.shape: (1,)
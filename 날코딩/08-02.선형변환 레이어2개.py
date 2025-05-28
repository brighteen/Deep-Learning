import numpy as np

from 날코딩.common.layers import Affine, MeanSquaredError
'''
data -> 선형 변환 1(node 2) -> 선형 변환 2(node 1) -> 손실함수
z = [1,1]
t = 0
'''

# 첫 선형변환 레이어: 두개의 예측값
# 두번째 선형변환 레이어: 하나의 예측값

np.random.seed(0)

z = np.array([1,1])
print(f"z: {z}, z.shape: {np.shape(z)}") # z: [1 1], z.shape: (2,)
z = np.reshape(z, (1, -1))
print(f"z: {z}, z.shape: {np.shape(z)}") # z: [[1 1]], z.shape: (1, 2)

t = np.array([0])
print(f"t: {t}, t.shape: {np.shape(t)}") # t: [0], t.shape: (1,)

W1 = np.random.randn(2,2)
# print(f"W1: {W1}, W1.shape: {np.shape(W1)}") # W1: [[1.76405235 0.40015721], [0.97873798 2.2408932 ]], W1.shape: (2, 2)
b1 = np.zeros(2,)
# print(f"b1: {b1}, b1.shape: {np.shape(b1)}") # b1: [0. 0.], b1.shape: (2,)
W2 = np.random.randn(2,1)
# print(f"W2: {W2}, W2.shape: {np.shape(W2)}") # W2: [[1.86755799], [0.95008842]], W2.shape: (2, 1)
b2 = np.zeros(1,)
# print(f"b2: {b2}, b2.shape: {np.shape(b2)}") # b2: [0.], b2.shape: (1,)

affine1 = Affine(W1, b1)
affine2 = Affine(W2, b2)
mse = MeanSquaredError()

print(f"W1: {affine1.W}, W1.shape: {np.shape(affine1.W)}") # W1: [[1.76405235 0.40015721], [0.97873798 2.2408932 ]], W1.shape: (2, 2)
print(f"b1: {affine1.b}, b1.shape: {np.shape(affine1.b)}") # b1: [0. 0.], b1.shape: (2,)
print(f"W2: {affine2.W}, W2.shape: {np.shape(affine2.W)}") # W2: [[1.86755799], [0.95008842]], W2.shape: (2, 1)
print(f"b2: {affine2.b}, b2.shape: {np.shape(affine2.b)}") # b2: [0.], b2.shape: (1,)

lr = 0.01

for i in range(1):
    print(f"\n==========={i+1}회차===========\n")
    # 순전파
    out = affine1.forward(z) # 첫번째 레이어 예측값 [[y^1_1], [y^1_2]]
    print(f"out[[y^1_1], [y^1_2]]: {out}, shape: {np.shape(out)}") # out: [[2.74279033 2.64105041]], out.shape: (1, 2)

    out = affine2.forward(out) # 두번째 레이어 예측값 y^2_1
    print(f"out[y^2_1]: {out}, shape: {np.shape(out)}") # out: [[2.54127985]], out.shape: (1, 1)

    out = mse.forward(out, t) # 손실값 Loss
    print(f"out(loss): {out}, shape: {np.shape(out)}") # out: 3.229051646341373, out.shape: ()

    print("============구분분==============")
    # 역전파
    dout = np.ones(1,) # dL/dL
    dout = mse.backward(dout) # dL/dy^2_1
    print(f"dout(dL/dy^2_1): {dout}, shape: {np.shape(dout)}") # dout: [[2.54127985]], dout.shape: (1, 1)

    dout = affine2.backward(dout) # dL/dy^1_1: [[dL/dy^1_1], [dL/dy^1_2]]
    print(f"dout([[dL/dy^1_1],[dL/dy_1_2]]): {dout}, shape: {np.shape(dout)}") # dout: [[4.74867999 2.41558857]], dout.shape: (1, 2)

    print(f"affine2.dW: {affine2.dW}, shape: {np.shape(affine2.dW)}") # affine2.dW: [[6.97019781] [6.71164819]], affine2.dW.shape: (2, 1)
    print(f"affine2.db: {affine2.db}, shape: {np.shape(affine2.db)}") # affine2.db: [2.54127985], affine2.db.shape: (1,)

    dout - affine1.backward(dout) # dL/dx: [[dL/dx^1_1], [dL/dx^1_2]]
    print(f"dout: {dout}, shape: {np.shape(dout)}") # dout: [[ 4.7459875  -2.48353659]], dout.shape: (1, 2)

    print(f"affine1.dW: {affine1.dW}, shape: {np.shape(affine1.dW)}") # affine1.dW: [[ 4.7459875  -2.48353659] [ 4.7459875  -2.48353659]], affine1.dW.shape: (2, 2)
    print(f"affine1.db: {affine1.db}, shape: {np.shape(affine1.db)}") # affine1.db: [ 4.7459875  -2.48353659], affine1.db.shape: (2,)

    # 가중치, 편향 업데이트
    affine2.W = affine2.W - lr * affine2.dW
    affine2.b = affine2.b - lr * affine2.db
    affine1.W = affine1.W - lr * affine1.dW
    affine1.b = affine1.b - lr * affine1.db
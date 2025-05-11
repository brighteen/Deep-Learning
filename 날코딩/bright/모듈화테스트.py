from Affine import LinearModel
from LossFunction import MeanSquaredError
import numpy as np

np.random.seed(0)
x = 2
t = 1

print("forward")
model = LinearModel()
mse = MeanSquaredError()

print(f"파라미터: {model.params}")
# 각 파라미터 출력
print(f"W1: {model.params['W1']}, b1: {model.params['b1']}")
print(f"W2: {model.params['W2']}, b2: {model.params['b2']}")

# 순전파 수행
y1, y2 = model.forward(x)
print(f"y_1: {y1}, y_2: {y2}")

# 손실 함수 계산
out = mse.forward(y2=y2, t=t)
print(f"loss: {out}")

print("\nbackward")
# 손실에 대한 기울기 계산
dout = mse.backward()
print(f"dL/dy2 = (y2 - t): {dout}")

# 역전파 수행 - LinearModel 클래스의 backward 메서드는 이제 (dx, grads) 형태로 반환
dx, grads = model.backward(dout=dout)

# 각 파라미터에 대한 기울기 출력
print(f"dW2: {grads['W2']}, db2: {grads['b2']}")
print(f"dW1: {grads['W1']}, db1: {grads['b1']}")
print(f"dx: {dx}")
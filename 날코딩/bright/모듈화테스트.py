from Affine import LinearModel
from LossFunction import MeanSquaredError
import numpy as np

np.random.seed(0)
x = 2
t = 1

print("forward")
model = LinearModel()
mse = MeanSquaredError()

print(f"w1: {model.w1}, b1: {model.b1}\nw2: {model.w2}, b2: {model.b2}")
# out = model.forward(x)
y1, y2 = model.forward(x)
print(f"y_1: {y1}, y_2: {y2}")
out = mse.forward(y2=y2, t=t)
print(f"loss: {out}")


print("\nbackward")
dout = mse.backward()
print(f"dL/dy2 = (y2 - t): {dout}")
dx, dw1, db1, dy1, dw2, db2 = model.backward(dout=dout)
print(f"dw2: {dw2}, db2: {db2}, dy1: {dy1}")
print(f"dw1: {dw1}, db1: {db1}, dx: {dx}")
import numpy as np

np.random.seed(0)

class LinearModel:
    def __init__(self):
        self.w = np.random.randn(1)
        self.b = np.zeros(1)
        self.x = None # forward에서의 x를 backward에서 갖다 써야하니까 x도 self를 취해줌
        # print(f"Initialize!\nw: {self.w}, b: {self.b}\n")
    
    def forward(self, x):
        self.x = x
        # print(f"Forward!\nx: {self.x}")
        return np.dot(self.w, self.x) + self.b # y = wx+b
    
    def backward(self, dout):
        # print(f'loss: {dout}')
        dw = np.dot(dout, self.x)
        db = np.dot(dout, 1)
        dx = np.dot(dout, self.w)
        # print(f"Backward\ndw: {dw}, db: {db}, dx: {dx}")
        return dw, db, dx
    
class MeanSquardError:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.t = t
        self.y = y
        return np.dot(0.5, (self.y - self.t)**2) # 0.5(y-t)^2
    
    def backward(self, dout=1):
        return np.dot((self.y - self.t), dout)


model = LinearModel()
mse = MeanSquardError()

print(f'w: {model.w}')
print(f'b: {model.b}')

x = 2
t = 1

# forward
out = model.forward(x)
print(f'[input x]: {x}, pred_y(wx+b): {out}')
out = mse.forward(out, t)
print(f'loss: {out}')

dout = mse.backward(dout=1.0)
print(f"dL/dy: {dout}")

dw, db, dx = model.backward(dout=dout)
print(f"dL/dw: {dw}, dL/db: {db}, dL/dx: {dx}")



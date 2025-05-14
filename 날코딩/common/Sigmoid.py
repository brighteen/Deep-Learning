import numpy as np

class Sigmoid:
    def __init__(self):
        self.z=None

    def forward(self, y):
        self.z = 1 / (1 + np.exp(-y))
        return self.z

    def backward(self, dout):
        return np.dot(self.z,(1-self.z)) * dout
    
if __name__ == "__main__":
    sigmoid = Sigmoid()

    # forward
    y = 100 # y가 0에서 멀어질수록 sigmoid의 기울기는 0에 가까워짐 -> 기울기 소실
    out = sigmoid.forward(y)
    print(f"pred_z: {out}")

    # backward
    dout = 1
    z_grad = sigmoid.backward(dout)
    print(f"z_grad: {z_grad}")
import numpy as np

class LinearModel:
    def __init__(self):
        self.w1 = np.random.randn(1)
        self.b1 = np.zeros(1)
        self.w2 = np.random.randn(1)
        self.b2 = np.zeros(1)
        self.x = None

    def forward(self, x):
        self.x = x
        self.y1 = np.dot(self.w1, self.x) + self.b1
        self.y2 = np.dot(self.w2, self.y1) + self.b2
        return self.y1, self.y2
    
    def backward(self, dout):
        dw2 = self.y1 * dout
        db2 = 1 * dout
        dy1 = self.w2 * dout

        dw1 = self.x * dy1
        db1 = 1 * dy1
        dx = self.w1 * dy1
        return dx, dw1, db1, dy1, dw2, db2
    
import numpy as np
np.random.seed(0)

class LinearModel:
    def init(self):
        self.w1 = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b1 = np.zeros(1)
        self.w2 = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b2 = np.zeros(1)
        self.x = None
        self.y1 = None
        self.y2 = None

    def forward(self, x):
        self.x = x
        self.y1 = np.dot(self.w1, self.x) + self.b1 # y1 = w1*x + b1
        self.y2 = np.dot(self.w2, self.y1) + self.b2 # y2 = w2*y1 + b2
        print(f'pred y: {self.y1}, y.shape: {self.y1.shape}')
        print(f'pred y: {self.y2}, y.shape: {self.y2.shape}')
        return self.y1, self.y2
    
    def backward(self, dout, y1, y2): # dout :아직 모르는 loss의 미분값
        # dout = dL_dy2
        dy1 = dout * self.w2
        dw2 = dout * self.y1
        db2 = dout 

        dx = dout * self.w1
        dw1 = 

        dw1 = np.dot(dy1, self.x)
        db1 = np.dot(dy1, 1)


        dx = np.dot(self.w1, dout) # dy_dx
        dw = np.d

        dy2dy1 = self.w2
        dy2dw2 = y1 * dout
        dy2db2 = dout
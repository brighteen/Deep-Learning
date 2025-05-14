import numpy as np

class MeanSquaredError:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.t = t
        self.y = y
        return np.dot(0.5, (self.y - self.t)**2) # 0.5(y-t)^2
    
    def backward(self, dout=1):
        return np.dot((self.y - self.t), dout)
        
if __name__ == "__main__": # 다른 파일에서 main.py를 가져올 때 이 __main__안에 코드는 무시
    np.random.seed(0)
    y = 4
    t = 1
    mse = MeanSquaredError()
    out = mse.forward(y, t)
    print(f'loss: {out}')
    dout = mse.backward(dout=1.0)
    print(f"dLdy: {dout}")
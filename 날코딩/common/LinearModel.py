import numpy as np

class LinearModel:
    def __init__(self):
        self.w = np.random.randn(1)
        self.b = np.zeros(1)
        self.x = None # forward에서의 x를 backward에서 갖다 써야하니까 x도 self를 취해줌
    
    def forward(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b # y = wx+b
    
    def backward(self, dout):
        # print(f'loss: {dout}')
        dw = np.dot(dout, self.x)
        db = np.dot(dout, 1)
        dx = np.dot(dout, self.w)
        return dw, db, dx
    
if __name__ == "__main__": # 다른 파일에서 main.py를 가져올 때 이 __main__안에 코드는 무시
    np.random.seed(0)
    model = LinearModel()
    print(f"w: {model.w}")
    print(f"b: {model.b}")

    # forward
    x = 2
    out = model.forward(x)
    print(f"y(wx + b): {out}")

    dw, db, dx = model.backward(dout=1)
    print(f"dL/dw: {dw}, dL/db: {db}, dL/dx: {dx}")
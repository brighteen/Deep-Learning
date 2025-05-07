import numpy as np

np.random.seed(42)

class LinearModel:
    def __init__(self):
        self.w = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b = np.zeros(1)
        self.x = None
        print("Initialize")
        print(f'weight w: {self.w}, w.shape: {self.w.shape}')
        print(f'bios b: {self.b}, b.shape: {self.b.shape}')

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.w,x) + self.b
        print("\nforward")
        print(f'input: {x}, pred: {self.y}')
        return self.y
    
    def backward(self, dout):
        # dout = dL_dy(손실함수loss의 미분값, 현재 1)
        dx = np.dot(dout, self.w) # dL_dx = dL_dy * dy_dx
        dw = np.dot(dout, self.x) # dL_dw = dL_dy * dy_dw
        db = np.dot(dout, 1) # dL_db = dL_dy * dy_db
        print("\nbackward")
        print(f"dout: {dout}, dx: {dx}, dw: {dw}, db: {db}")

        return dx, dw, db # dx: x에 대한 미분값, dw: w에 대한 미분값, db: b에 대한 미분값


x = 2
model = LinearModel() # 모델 선언
print(f"w: {model.w}, b: {model.b}")

y = model.forward(x)
print(f"print y: {model.y}")

dout = 1
grad_x, grad_w, grad_b = model.backward(dout)
# print(f'grad_x(w): {grad_x}, grad_w(x): {grad_w}, grad_b(1 * dout): {grad_b}')
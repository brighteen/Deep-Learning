import numpy as np

'''
def 내적(a, b):
    return np.dot(a, b)
a = 1
b =2 
c = 3
print(f'a + b = {내적(1, 2)+ 3}')
print(np.dot(a, b)+ c)
print(f'{a} * {b} + {c} = {a*b + c}')
'''

np.random.seed(0)

class LinearModel:
    def __init__(self):
        self.w = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b = np.zeros(1)
        self.x = None
        self.y = None
        print("Initialize!")
        print(f'weight w: {self.w}, w.shape: {self.w.shape}')
        print(f'bios b: {self.b}, b.shape: {self.b.shape}')

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.w,x) + self.b
        # print(f'input x: {x}')
        # print(f'pred y: {y}, y.shape: {y.shape}')
        print("\nforward")
        print(f'input: {model.x}, pred: {self.y}')
        return self.y
    
    def backward(self, dout): # dout :아직 모르는 loss의 미분값
        # dout = dL_dy
        dx = np.dot(dout, self.w) # dL_dx = dL_dy * dy_dx
        dw = np.dot(dout, self.x) # dL_dw = dL_dy * dy_dw
        db = np.dot(dout, 1) # dL_db = dL_dy * dy_db
        print("\nbackward")

        return dx, dw, db # dx: x에 대한 미분값, dw: w에 대한 미분값, db: b에 대한 미분값


x = 2
model = LinearModel() # 모델 선언
# print(f'w: {model.w}')
# print(f'b: {model.b}')
y = model.forward(x)
# print(x) # 전역변수 x를 출력
# print(f'\ninput: {model.x}, pred: {y}') # model 안에 x를 출력

dout = 1
grad_x, grad_w, grad_b = model.backward(dout)
print(f'grad_x(w): {grad_x}, grad_w(x): {grad_w}, grad_b(1 * dout): {grad_b}')
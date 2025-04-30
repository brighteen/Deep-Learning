import numpy as np
np.random.seed(0)

class LinearModel:
    def __init__(self):
        self.w1 = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b1 = np.zeros(1)
        self.w2 = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b2 = np.zeros(1)
        self.x = None
        self.y1 = None
        self.y2 = None
        print(f'[debug] w1: {self.w1}, b1: {self.b1}, w2: {self.w2}, b2: {self.b2}')
        print(f'[debug] x: {self.x}, y1: {self.y1}, y2: {self.y2}')

    def forward(self, x):
        self.x = x
        self.y1 = np.dot(self.w1, self.x) + self.b1 # y1 = w1*x + b1
        self.y2 = np.dot(self.w2, self.y1) + self.b2 # y2 = w2*y1 + b2
        print(f'[debug] x: {self.x}, y1: {self.y1}, y2: {self.y2}')
        return self.y1, self.y2
    
    def backward(self, dout): # dout: loss의 미분값
        # dout = dL/dy2
        dw2 = self.y1 * dout # w2가 움직일때 Loss의 변화량 dL/dw2 = dy2/dw2 * dL/dy2
        db2 = 1 * dout # dL/db2 = dy2/db2 * dL/dy2
        dy1 = self.w2 * dout # dL/dy1 = dy2/dy1 * dL/dy2
        print(f'[debug] dw2: {dw2}, db2: {db2}, dy1: {dy1}')

        dw1 = self.x * dy1 # dL/dw1 = dy1/dw1 * dy2/dy1 * dL/dy2 = dy1/dw1 * dy1
        db1 = 1 * dy1 # dL/db1 = dy1/db1 * dy1
        dx = self.w1 * dy1 # dL/dx = dy1/dx * dy1, 이건 안구해도 됨(왜냐면 x는 입력값이니까 미분값을 건네줄 애가 없음)
        print(f'[debug] dw1: {dw1}, db1: {db1}, dx: {dx}')
        return dx, dw1, db1, dy1, dw2, db2
    

print('\n---\n')
x = 2
dout = 1 # 
model = LinearModel() # 모델 선언
print('\n---\n')
y1, y2 = model.forward(x)
# print(f'\ninput: {model.x}, pred y1: {y1}, pred y2: {y2}')
print('\n---\n')
grad_x, grad_w1, grad_b1, grad_y1, grad_w2, grad_b2 = model.backward(1)
# print(f'\ndout: {dout}')
# print(f'\n[layer 1] \ndx: {grad_x}, dw1: {grad_w1}, db1: {grad_b1}')
# print(f'\n[layer 2] \ndy1: {grad_y1}, dw2= y1: {grad_w2}, db2: {grad_b2}')
import numpy as np

np.random.seed(1)

class LinearModel: # 한번의 선형변환 구현
    def __init__(self):
        self.w = np.random.randn(1) # w: 표준정규분포에서 랜덤하게 초기화
        self.b = np.zeros(1)
        self.x = None
        print(f'\n[Model] Initialize\tweight w: {self.w}, w.shape: {self.w.shape}\tbios b: {self.b}, b.shape: {self.b.shape}')

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.w,x) + self.b
        print(f'\n[Model] Forward\t\tinput: {x}, pred y: {self.y}')
        return self.y
    
    def backward(self, dout):
        # dout = dL_dy(손실함수loss의 미분값, 현재 1)
        dx = np.dot(dout, self.w) # dL_dx = dL_dy * dy_dx
        dw = np.dot(dout, self.x) # dL_dw = dL_dy * dy_dw
        db = np.dot(dout, 1) # dL_db = dL_dy * dy_db
        print(f"\n[Model] Backward\tdout: {dout}, dx: {dx}, dw: {dw}, db: {db}")

        return dx, dw, db # dx: x에 대한 미분값, dw: w에 대한 미분값, db: b에 대한 미분값
    
    # def mean_squared_error(self, t):
    #     mse = 0.5 * np.sum((self.y - t)**2)
    #     print(f"mse: {mse}")
    #     return mse

class MeanSquaredError:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
        print(f"\n[MSE] Initialize!\tpred y: {self.y}, t: {t}")

    def forward(self, y, t):
        self.y = y
        self.t = t
        self.loss = 0.5 * (self.y - self.t)**2
        print(f"\n[MSE] Forward\t\ty: {self.y}, t: {self.t}, loss: {loss}")
        return self.loss
    
    def backward(self, dout=1):
        dL_dy = (self.y - self.t) * dout
        print(f"\n[MSE] Backward\t\tdL_dy: {dL_dy}")
        return dL_dy # y - t 반환


x = 2
t = 1
lr = 0.01

model = LinearModel()
mse = MeanSquaredError()

for i in range(1):
    print(f"\n반복 횟수 {i+1}")
    # forward
    out = model.forward(x)
    out = mse.forward(y=out, t=t)

    # backward
    dout = mse.backward() # dL/dy
    print(f"\n[update 전] \tw: {model.w}, b: {model.b}")
    dx, dw, db = model.backward(dout=dout)

    # update
    model.w = model.w - lr * (dx) # dx = dLdy * dy/dx
    model.b = model.b - lr * (db) # db = dL/dy * dy/db
    print(f"\n[update 후] \tw: {model.w}, b: {model.b}")


'''
[프로세스]
x = 2
t = 1
lr = 0.01
model = LinearModel()
mse = MeanSquaredError()
print(f"w,b: {model.w, model.b}") # 초기 w, b

out = model.forward(x)
print(out)

out = mse.forward(y=out, t=t)
print(out)

dout = mse.backward() # dL/dy
print(dout)

dx, dw, db = model.backward(dout=dout) # dL/dx

out = model.forward(x)
print(out)

out = mse.forward(y=out, t=t)
print(out)
'''

'''
[반복문 프로세스]
x = 2
t = 1
lr = 0.01
model = LinearModel()
mse = MeanSquaredError()

for i in range(100):
    # forward
    out = model.forward(x)
    print(f"{i}번째 전파 출력값 : {out})

    out = mse.forward(y=out, t=t)
    print(f"{i}번째 loss값 : {out})

    # backward
    dout = mse.backward() # dL/dy
    dx, dw, db = model.backward(dout=dout)

    model.w -= lr * dw
    model.b -= lr * db

'''
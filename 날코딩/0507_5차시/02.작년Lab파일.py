import numpy as np

class LinearModel:
    def __init__(self, W, b):
        self.W = W # 가중치 행렬
        self.b = b # 편향(바이어스(b) 벡터)

        self.original_x_shape = None # 입력 데이터의 원래 형태를 저장
        self.x = None # 입력 데이터를 저장 (역전파에 필요)

        # 가중치, 편향에 대한 기울기 계산(역전파에서 계산)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape # 입력 데이터의 원래 형태 저장
        x = x.reshape(x.shape[0],-1) # 입력 데이터를 2차원 형태로 변환
        self.x = x # 변환된 입력 데이터를 저장

        out = np.dot(self. x, self.W) + self.b # x에 가중치(내적)와 편향을 적용 out = X*W + b
        return out

    def backward(self, dout):
        dx = np.dot(dout,self.W.T) # 입력값 x에 대한 기울기 계산
        # why? 은닉층이 여러 개일 경우 입력값이 이전 층에서 역전파를 전달받기 위한 매개체 역할
        # 은닉층이 하나일때는 고려하지 않음.

        # 가중치, 편향에 대한 기울기 계산(역전파에서 계산)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # 입력값의 원래 모양 복원원
        return dx
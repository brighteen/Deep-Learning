import numpy as np

class Affine:
    """
    Affine 변환 레이어: 신경망에서 완전연결(fully-connected) 계층을 구현
    y = x*W + b 형태의 선형 변환을 수행
    """
    def __init__(self, W, b):
        """
        Affine 레이어 초기화
        
        매개변수:
            W: 가중치 행렬
            b: 편향 벡터
        """
        self.W = W  # 가중치 행렬 저장
        self.b = b  # 편향 벡터 저장
        
        self.x = None  # 순전파 시 입력 데이터를 저장할 변수
        self.original_x_shape = None  # 입력 데이터의 원래 형상 저장 (텐서 처리 위함)
        # 가중치와 편향 매개변수의 미분
        self.dW = None  # 가중치에 대한 기울기 저장 변수
        self.db = None  # 편향에 대한 기울기 저장 변수

    def forward(self, x):
        """
        순전파 계산: y = x*W + b
        
        매개변수:
            x: 입력 데이터
            
        반환값:
            out: Affine 변환 결과
        """
        # 텐서 대응 - 입력 데이터의 형상 변환
        self.original_x_shape = x.shape  # 원래 입력 형상 저장
        x = x.reshape(x.shape[0], -1)  # 2차원 형태로 변환 (배치크기, 입력차원)
        self.x = x  # 변환된 입력 저장

        out = np.dot(self.x, self.W) + self.b  # Affine 변환 계산: y = x*W + b

        return out

    def backward(self, dout):
        """
        역전파 계산: 입력, 가중치, 편향에 대한 기울기 계산
        
        매개변수:
            dout: 출력에 대한 기울기
            
        반환값:
            dx: 입력 x에 대한 기울기
        """
        dx = np.dot(dout, self.W.T)  # 입력 x에 대한 기울기: dx = dout*W^T
        self.dW = np.dot(self.x.T, dout)  # 가중치 W에 대한 기울기: dW = x^T*dout
        # [[dL/dw11],[dL/dw21]] = [[dy/dw11*dL/dy],[dy/dw21*dL/dy]] -> np.dot으로 손쉽게 행렬계산
        self.db = np.sum(dout, axis=0)  # 편향 b에 대한 기울기: db = sum(dout)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 원래 모양으로 변경(텐서 대응)
        return dx


class MeanSquaredError:
    """
    평균 제곱 오차(MSE) 손실 함수 구현
    손실 함수: L = 0.5 * mean((y - t)^2)
    """
    def __init__(self):
        self.loss = None  # 계산된 손실 값을 저장하는 변수
        self.y = None     # 신경망의 예측값을 저장하는 변수
        self.t = None     # 훈련 데이터의 정답값을 저장하는 변수

    def forward(self, y, t):
        """
        순전파: MSE 손실값 계산
        
        매개변수:
            y: 신경망의 출력(예측값)
            t: 정답 레이블
            
        반환값:
            self.loss: 계산된 MSE 손실값
        """
        self.y = y  # 예측값 저장
        self.t = t  # 정답값 저장
        
        # 오차 계산: 0.5 * 평균((예측값 - 정답값)^2)
        self.loss = 0.5 * np.mean((self.y - self.t) ** 2)
        return self.loss

    def backward(self, dout=1):
        """
        역전파: 예측값에 대한 손실 함수의 기울기 계산
        
        매개변수:
            dout: 출력에 대한 기울기 (기본값 1)
            
        반환값:
            dx: 입력(예측값 y)에 대한 기울기
        """
        batch_size = self.t.shape[0]  # 배치 크기 계산
        
        # 예측값 y에 대한 기울기 계산: d(MSE)/dy = (y - t) / batch_size
        # y가 t보다 크면 양수, 작으면 음수 기울기가 됨
        dx = (self.y - self.t) * dout / batch_size
        return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        # out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
    
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
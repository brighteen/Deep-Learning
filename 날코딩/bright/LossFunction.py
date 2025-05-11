import numpy as np  # 수치 계산을 위한 NumPy 라이브러리 import

class MeanSquaredError:
    """
    평균 제곱 오차(Mean Squared Error) 손실 함수 클래스
    L = 0.5 * (y - t)^2 형태의 손실 함수를 구현합니다.
    """
    def __init__(self):
        """
        손실 함수의 변수 초기화
        """
        self.cache = {}  # 중간 값을 저장할 캐시 딕셔너리
        self.loss = None  # 손실 값 저장 변수

    def forward(self, y2, t):
        """
        순전파(forward propagation)를 통해 손실 값 계산
        
        Parameters:
            y2: 모델의 예측 출력값
            t: 정답(타겟) 값
            
        Returns:
            loss: 손실 함수 값 (0.5 * (y2 - t)^2)
        """
        # 역전파 계산을 위해 입력 값을 캐시에 저장
        self.cache = {
            'y2': y2,  # 모델의 출력값
            't': t     # 정답 값
        }
        
        # 평균 제곱 오차 계산: L = 0.5 * (y - t)^2
        self.loss = 0.5 * (y2 - t)**2
        
        return self.loss
    
    def backward(self, dout=1):
        """
        역전파(backward propagation)를 통해 기울기 계산
        
        Parameters:
            dout: 출력에 대한 기울기 (기본값은 1)
            
        Returns:
            dy2: y2에 대한 손실 함수의 기울기 (∂L/∂y2 = (y2 - t))
        """
        y2 = self.cache['y2']
        t = self.cache['t']
        
        # 평균 제곱 오차의 y2에 대한 미분: ∂L/∂y2 = (y2 - t)
        dy2 = (y2 - t) * dout
        
        return dy2
# NumPy 라이브러리를 가져옵니다. 행렬 연산과 수학 함수를 사용하기 위함입니다.
import numpy as np

# 타입 힌트를 위한 typing 모듈에서 필요한 타입들을 가져옵니다.
from typing import Tuple, Optional, Union

# Affine 변환을 수행하는 레이어 클래스를 정의합니다.
class Affine:
    """
    Affine 변환 레이어 클래스
    선형 변환 (y = Wx + b)을 수행
    """
    # 클래스 초기화 메서드입니다.
    def __init__(self, input_size: int, output_size: int, weight_init_std: float = 0.01):
        """
        Affine 레이어 초기화
        
        매개변수:
            input_size (int): 입력 데이터의 크기
            output_size (int): 출력 데이터의 크기
            weight_init_std (float): 가중치 초기화 표준편차
        """
        # 가중치 행렬 W를 랜덤하게 초기화합니다. 크기는 (입력 크기, 출력 크기)입니다.
        # 표준 정규 분포에서 샘플링한 값에 표준편차를 곱합니다.
        self.W = weight_init_std * np.random.randn(input_size, output_size)
        
        # 편향 벡터 b를 0으로 초기화합니다. 크기는 (출력 크기)입니다.
        self.b = np.zeros(output_size)
        
        # 역전파 계산에 필요한 중간값들을 저장할 변수들을 초기화합니다.
        self.x = None  # 입력 데이터 저장
        self.dW = None  # 가중치에 대한 기울기
        self.db = None  # 편향에 대한 기울기

    # 순전파(forward propagation) 연산을 수행하는 메서드입니다.
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        순전파 계산
        
        매개변수:
            x (np.ndarray): 입력 데이터, 형상 (배치 크기, 입력 크기)
            
        반환값:
            np.ndarray: 출력 데이터, 형상 (배치 크기, 출력 크기)
        """
        # 입력 데이터를 인스턴스 변수에 저장 (역전파 계산에 사용)
        self.x = x
        
        # 선형 변환 y = Wx + b 수행
        out = np.dot(x, self.W) + self.b
        
        # 변환 결과 반환
        return out
        
    # 역전파(backward propagation) 연산을 수행하는 메서드입니다.
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        역전파 계산
        
        매개변수:
            dout (np.ndarray): 출력에 대한 기울기, 형상 (배치 크기, 출력 크기)
            
        반환값:
            np.ndarray: 입력에 대한 기울기, 형상 (배치 크기, 입력 크기)
        """
        # dout: 출력에 대한 손실 함수의 기울기 (dL/dy)
        
        # 입력 x에 대한 기울기 계산
        # dx = dout * W^T (행렬 곱)
        dx = np.dot(dout, self.W.T)
        
        # 가중치 W에 대한 기울기 계산
        # dW = x^T * dout (행렬 곱)
        self.dW = np.dot(self.x.T, dout)
        
        # 편향 b에 대한 기울기 계산
        # db = dout의 각 열에 대한 합 (배치 축으로 합산)
        self.db = np.sum(dout, axis=0)
        
        # 입력에 대한 기울기 반환
        return dx


# 2층으로 구성된 신경망 모델 클래스를 정의합니다.
class TwoLayerNet:
    """
    2층 신경망 모델
    """
    # 클래스 초기화 메서드입니다.
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        2층 신경망 모델 초기화
        
        매개변수:
            input_size (int): 입력 크기
            hidden_size (int): 은닉층 크기
            output_size (int): 출력 크기
        """
        # 첫 번째 Affine 레이어 초기화 (입력층 -> 은닉층)
        self.layer1 = Affine(input_size, hidden_size)
        
        # 두 번째 Affine 레이어 초기화 (은닉층 -> 출력층)
        self.layer2 = Affine(hidden_size, output_size)
        
        # 모든 레이어를 리스트에 저장 (학습 시 반복 처리를 위함)
        self.layers = [self.layer1, self.layer2]
        
    # 순전파 연산을 수행하는 메서드입니다.
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        순전파 계산
        
        매개변수:
            x (np.ndarray): 입력 데이터
            
        반환값:
            np.ndarray: 출력 데이터
        """
        # 첫 번째 레이어의 순전파 계산 (입력 -> 은닉층)
        y1 = self.layer1.forward(x)
        
        # 두 번째 레이어의 순전파 계산 (은닉층 -> 출력)
        y2 = self.layer2.forward(y1)
        
        # 중간층과 출력층의 값을 반환
        return y1, y2
        
    # 역전파 연산을 수행하는 메서드입니다.
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        역전파 계산
        
        매개변수:
            dout (np.ndarray): 출력에 대한 기울기
            
        반환값:
            np.ndarray: 입력에 대한 기울기 및 각 매개변수에 대한 기울기
        """
        # 출력층에서 은닉층으로 역전파 (레이어2)
        dy1 = self.layer2.backward(dout)
        
        # 은닉층에서 입력층으로 역전파 (레이어1)
        dx = self.layer1.backward(dy1)
        
        # 각 레이어의 매개변수에 대한 기울기들을 모아서 반환
        return dx, self.layer1.dW, self.layer1.db, dy1, self.layer2.dW, self.layer2.db
    
    # 학습(매개변수 업데이트)을 수행하는 메서드입니다.
    def update(self, learning_rate: float = 0.01) -> None:
        """
        매개변수 업데이트 (학습)
        
        매개변수:
            learning_rate (float): 학습률
        """
        # 각 레이어에 대해 반복
        for layer in self.layers:
            # 가중치 업데이트: W = W - learning_rate * dW
            layer.W -= learning_rate * layer.dW
            
            # 편향 업데이트: b = b - learning_rate * db
            layer.b -= learning_rate * layer.db
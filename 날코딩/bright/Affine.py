import numpy as np  # 수치 계산을 위한 NumPy 라이브러리 import

class LinearModel:
    """
    두 개의 Affine 변환(선형 변환)을 연결한 간단한 선형 모델 클래스
    이 모델은 y = W2(W1x + b1) + b2 형태의 계산을 수행합니다.
    """
    def __init__(self):
        """
        모델 초기화: 가중치와 편향을 초기화하고 중간 계산값을 저장할 캐시 생성
        """
        # 가중치와 편향을 딕셔너리로 관리 (유지보수와 확장성 향상)
        self.params = {
            'W1': np.random.randn(1),  # 첫 번째 층의 가중치를 정규분포에서 랜덤 초기화
            'b1': np.zeros(1),         # 첫 번째 층의 편향을 0으로 초기화
            'W2': np.random.randn(1),  # 두 번째 층의 가중치를 정규분포에서 랜덤 초기화
            'b2': np.zeros(1)          # 두 번째 층의 편향을 0으로 초기화
        }
        self.x = None                  # 입력값을 저장할 변수
        self.cache = {}                # 역전파 계산에 필요한 중간값들을 저장할 캐시
    
    def forward(self, x):
        """
        순전파(forward propagation) 수행
        
        Parameters:
            x: 입력값
        
        Returns:
            y1: 첫 번째 Affine 변환의 출력
            y2: 두 번째 Affine 변환의 출력 (최종 출력)
        """
        self.x = x                     # 입력값 저장
        
        # 첫 번째 Affine 변환: y1 = W1x + b1
        y1 = np.dot(self.params['W1'], x) + self.params['b1']
        
        # 두 번째 Affine 변환: y2 = W2y1 + b2
        y2 = np.dot(self.params['W2'], y1) + self.params['b2']
        
        # 역전파 계산을 위해 중간 계산값들을 캐시에 저장
        self.cache = {
            'x': x,       # 입력값
            'y1': y1,     # 첫 번째 층의 출력
            'y2': y2      # 두 번째 층의 출력
        }
        
        return y1, y2     # 각 층의 출력 반환
    
    def backward(self, dout):
        """
        역전파(backward propagation) 수행 - 기울기(gradient) 계산
        
        Parameters:
            dout: 출력에 대한 기울기 (∂L/∂y2, L은 손실함수)
        
        Returns:
            dx: 입력 x에 대한 기울기 (∂L/∂x)
            grads: 각 가중치와 편향에 대한 기울기를 담은 딕셔너리
        """
        # 캐시에서 순전파 때 저장한 중간값들을 가져옴
        x, y1 = self.cache['x'], self.cache['y1']
        
        # 각 파라미터에 대한 기울기를 저장할 딕셔너리
        grads = {}
        
        # 두 번째 층의 역전파 계산
        grads['W2'] = y1 * dout        # W2에 대한 기울기: ∂L/∂W2 = y1 * dout
        grads['b2'] = 1 * dout         # b2에 대한 기울기: ∂L/∂b2 = dout
        dy1 = self.params['W2'] * dout # y1에 대한 기울기: ∂L/∂y1 = W2 * dout
        
        # 첫 번째 층의 역전파 계산
        grads['W1'] = x * dy1          # W1에 대한 기울기: ∂L/∂W1 = x * dy1
        grads['b1'] = 1 * dy1          # b1에 대한 기울기: ∂L/∂b1 = dy1
        dx = self.params['W1'] * dy1   # x에 대한 기울기: ∂L/∂x = W1 * dy1
        
        return dx, grads   # 입력에 대한 기울기와 파라미터 기울기 딕셔너리 반환
    
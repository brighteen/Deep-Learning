import numpy as np

# 선형변환이 2번 있고 MSE 손실함수를 사용하는 직접 구현 모델
class LinearModel:
    def __init__(self):
        # 딕셔너리로 파라미터 표현
        # 가중치는 정규분포(randn)로, 편향은 zeros로 초기화
        self.params = {
            'w1': np.random.randn(1, 1),  # 첫 번째 선형변환 가중치 (정규분포로 초기화)
            'b1': np.zeros((1, 1)),       # 첫 번째 선형변환 편향 (0으로 초기화)
            'w2': np.random.randn(1, 1),  # 두 번째 선형변환 가중치 (정규분포로 초기화)
            'b2': np.zeros((1, 1))        # 두 번째 선형변환 편향 (0으로 초기화)
        }
        self.grads = {
            'w1': None,
            'b1': None,
            'w2': None,
            'b2': None
        }
        self.x = None
        self.y1 = None
        self.y2 = None
        
    def forward(self, x):
        """순전파(forward) 구현"""
        self.x = x
        self.y1 = np.dot(self.params['w1'], x) + self.params['b1']  # 첫 번째 선형변환
        self.y2 = np.dot(self.params['w2'], self.y1) + self.params['b2']  # 두 번째 선형변환
        return self.y1, self.y2
    
    def backward(self, dout):
        """역전파(backward) 구현"""
        # dout: 손실함수에서 전달된 기울기(dL/dy2)
        
        # 두 번째 선형변환에 대한 기울기
        self.grads['w2'] = self.y1 * dout
        self.grads['b2'] = 1.0 * dout
        dy1 = self.params['w2'] * dout  # dy1 = dL/dy1 = dL/dy2 * dy2/dy1
        
        # 첫 번째 선형변환에 대한 기울기
        self.grads['w1'] = self.x * dy1
        self.grads['b1'] = 1.0 * dy1
        dx = self.params['w1'] * dy1  # dx = dL/dx = dL/dy1 * dy1/dx
        
        return dx
    
    def compute_loss(self, y, t):
        """평균 제곱 오차(MSE) 계산"""
        return 0.5 * (y - t)**2  # 0.5를 곱하는 이유는 미분 시 계산을 간단하게 하기 위함
    
    def compute_loss_gradient(self, y, t):
        """손실함수의 기울기 계산: dL/dy = (y-t)"""
        return y - t

# 데이터와 타겟 설정
x = 2.0  # 입력값 x = 2
t = 1.0  # 목표값 t = 1

# 모델 인스턴스 생성
model = LinearModel()

# 학습 과정
learning_rate = 0.1
epochs = 5

print("=== 학습 시작 ===")
for epoch in range(epochs):
    # 순전파
    y1, y2 = model.forward(x)
    loss = model.compute_loss(y2, t)
      # 손실함수의 기울기 계산
    dout = model.compute_loss_gradient(y2, t)
    
    # 역전파
    dx = model.backward(dout)
      # 파라미터 출력
    if epoch == 0:
        print(f"\n== 초기 상태 ==")
        print(f"입력 x: {x}, 목표값 t: {t}")
        # 넘파이 배열의 단일 원소에 명시적으로 접근
        print(f"y1: {y1[0][0]:.4f}, y2: {y2[0][0]:.4f}, 손실: {loss[0][0]:.4f}")
        print("\n파라미터와 기울기:")
        for key in model.params.keys():
            # 넘파이 배열의 단일 원소에 명시적으로 접근
            print(f"  {key}: {model.params[key][0][0]:.4f}, grad: {model.grads[key][0][0]:.4f}")
    
    # 파라미터 업데이트 (경사하강법)
    for key in model.params:
        model.params[key] -= learning_rate * model.grads[key]
      # 현재 상태 출력
    print(f"\n== 에폭 {epoch+1} ==")
    print(f"y2 예측값: {y2[0][0]:.4f}, 손실: {loss[0][0]:.4f}")
    print("주요 파라미터:")
    for key in model.params:
        print(f"  {key}: {model.params[key][0][0]:.4f}")

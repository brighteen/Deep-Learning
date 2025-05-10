import torch
import torch.nn as nn

# 선형변환이 2번 있고 MSE 손실함수를 사용하는 간소화된 모델
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 두 개의 선형 변환 레이어 정의
        self.linear1 = nn.Linear(1, 1)  # 첫 번째 선형변환
        self.linear2 = nn.Linear(1, 1)  # 두 번째 선형변환
        
        # 가중치 초기화 (간단한 값으로 설정)
        self.linear1.weight.data.fill_(1.0)
        self.linear1.bias.data.fill_(0.0)
        self.linear2.weight.data.fill_(1.0)
        self.linear2.bias.data.fill_(0.0)
        
    def forward(self, x):
        # 순전파(forward) 구현
        y1 = self.linear1(x)  # 첫 번째 선형변환
        y2 = self.linear2(y1)  # 두 번째 선형변환
        return y1, y2

# 데이터와 타겟 설정
x = torch.tensor([[2.0]], requires_grad=True)  # 입력값 x = 2
target = torch.tensor([[1.0]])  # 목표값 t = 1

# 모델과 손실함수 설정
model = LinearModel()
criterion = nn.MSELoss()

# 학습 과정
learning_rate = 0.1
epochs = 5

print("=== 학습 시작 ===")
for epoch in range(epochs):
    # 순전파
    y1, y2 = model(x)
    loss = criterion(y2, target)
    
    # 역전파
    model.zero_grad()
    loss.backward()
    
    # 파라미터 출력
    if epoch == 0:
        print(f"\n== 초기 상태 ==")
        print(f"입력 x: {x.item()}, 목표값 t: {target.item()}")
        print(f"y1: {y1.item():.4f}, y2: {y2.item():.4f}, 손실: {loss.item():.4f}")
        print("\n파라미터와 기울기:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.data.item():.4f}, grad: {param.grad.item():.4f}")
    
    # 파라미터 업데이트
    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate * param.grad
    
    # 현재 상태 출력
    print(f"\n== 에폭 {epoch+1} ==")
    print(f"y2 예측값: {y2.item():.4f}, 손실: {loss.item():.4f}")
    print("주요 파라미터:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.data.item():.4f}")

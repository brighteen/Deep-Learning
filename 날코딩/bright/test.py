import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        
        # 레이어 정의 (내부적으로 Parameter 객체로 변환됨)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 모델 생성
model = SimpleModel(1, 2, 1)

print(f"Model: {model}")
print(f"\nModel type: {type(model)}")
print(f"\nModel parameters: {model.parameters()}")
print(f"\nModel parameters type: {type(model.parameters())}")

# 모델의 파라미터 수
print(f"\nModel parameters count: {sum(p.numel() for p in model.parameters())}")

# 각 레이어의 파라미터 수
print(f"-----------\nLayer 1 parameters count: {model.fc1.weight.numel() + model.fc1.bias.numel()}")
print(f"Layer 2 parameters count: {model.fc2.weight.numel() + model.fc2.bias.numel()}")

for name, param in model.named_parameters():
    print(f"\nLayer: {name} | Size: {param.size()} | Values: {param[:2]} \n")

# # 특정 파라미터 직접 접근하기
print(f"Weight of first layer: {model.fc1.weight.shape}")
print(f"Bias of first layer: {model.fc1.bias.shape}")
print(f"Weight of second layer: {model.fc2.weight.shape}")
print(f"Bias of second layer: {model.fc2.bias.shape}")

x = 2
x = nn.tensor([[x]], dtype=nn.float32)
print(f"\nInput tensor: {x}")
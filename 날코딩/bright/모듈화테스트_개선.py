"""
간결하고 명확한 순전파-역전파-업데이트 과정 데모
"""
import numpy as np
from Affine import LinearModel
from LossFunction import MeanSquaredError
import sys

# 에러 디버깅을 위한 코드
print("Python version:", sys.version)
print("NumPy version:", np.__version__)

def print_section(title):
    """섹션 타이틀을 출력하는 함수"""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

def print_dict(title, d, indent=2):
    """딕셔너리 내용을 보기 좋게 출력하는 함수"""
    print(f"\n{' ' * indent}■ {title}:")
    for k, v in d.items():
        print(f"{' ' * (indent+2)}● {k}: {v}")

def main():
    # 1. 초기 설정
    np.random.seed(0)  # 재현성을 위한 시드 설정
    x = 2              # 입력값
    t = 1              # 타겟(정답)값
    learning_rate = 0.1  # 학습률

    # 2. 모델 및 손실 함수 초기화
    model = LinearModel()
    mse = MeanSquaredError()

    # 3. 초기 파라미터 출력
    print_section("1. 초기 파라미터")
    print_dict("모델 파라미터", model.params)
    
    # 4. 순전파 (Forward Propagation)
    print_section("2. 순전파 (Forward Propagation)")
    print(f"  입력값 (x): {x}")
    print(f"  타겟값 (t): {t}")
    
    y1, y2 = model.forward(x)
    print(f"\n  ▶ 첫 번째 층 출력 (y1): {y1}")
    print(f"  ▶ 두 번째 층 출력 (y2): {y2}")
    
    # 5. 손실 계산
    loss = mse.forward(y2, t)
    print(f"\n  ▶ 손실 함수 값 (MSE): {loss}")
    
    # 6. 역전파 (Backward Propagation)
    print_section("3. 역전파 (Backward Propagation)")
    
    # 손실에 대한 출력 기울기 계산 (∂L/∂y2)
    dy2 = mse.backward()
    print(f"  출력층 기울기 (∂L/∂y2): {dy2}")
    
    # 모델 파라미터에 대한 기울기 계산
    dx, grads = model.backward(dy2)
    print_dict("파라미터별 기울기", grads)
    print(f"\n  ▶ 입력에 대한 기울기 (∂L/∂x): {dx}")
    
    # 7. 파라미터 업데이트
    print_section("4. 파라미터 업데이트 (learning_rate = 0.1)")
    old_params = model.params.copy()
    
    # W1 업데이트
    model.params['W1'] -= learning_rate * grads['W1']
    # b1 업데이트
    model.params['b1'] -= learning_rate * grads['b1']
    # W2 업데이트
    model.params['W2'] -= learning_rate * grads['W2']
    # b2 업데이트
    model.params['b2'] -= learning_rate * grads['b2']
    
    print("  ■ 업데이트 전후 비교:")
    for key in model.params:
        old_val = old_params[key][0] if isinstance(old_params[key], np.ndarray) else old_params[key]
        new_val = model.params[key][0] if isinstance(model.params[key], np.ndarray) else model.params[key]
        change = new_val - old_val
        print(f"    ● {key}: {old_val:.6f} → {new_val:.6f} (변화량: {change:.6f})")
    
    # 8. 업데이트 후 모델 평가
    print_section("5. 업데이트 후 모델 평가")
    y1_new, y2_new = model.forward(x)
    loss_new = mse.forward(y2_new, t)
    
    # NumPy 배열을 스칼라로 변환하여 출력
    y2_val = y2[0] if isinstance(y2, np.ndarray) else y2
    y2_new_val = y2_new[0] if isinstance(y2_new, np.ndarray) else y2_new
    loss_val = loss[0] if isinstance(loss, np.ndarray) else loss
    loss_new_val = loss_new[0] if isinstance(loss_new, np.ndarray) else loss_new
    
    print(f"  ■ 이전 출력: {y2_val:.6f}")
    print(f"  ■ 새로운 출력: {y2_new_val:.6f}")
    print(f"  ■ 타겟값: {t}")
    print(f"\n  ■ 이전 손실: {loss_val:.6f}")
    print(f"  ■ 새로운 손실: {loss_new_val:.6f}")
    print(f"  ■ 손실 감소: {loss_val - loss_new_val > 0}")
    print(f"  ■ 손실 변화량: {loss_val - loss_new_val:.6f}")

if __name__ == "__main__":
    main()

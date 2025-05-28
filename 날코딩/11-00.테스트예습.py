# 함수·레이어·최적화 기법 등 필요한 모듈을 한 번에 가져오기
from common.functions               import *
from common.gradient                import *
from common.layers                  import *
from common.multi_layer_net         import *
from common.multi_layer_net_extend  import *
from common.optimizer               import *
from common.trainer                 import *
from common.util                    import *

import numpy as np
'''
XOR 문제를 신경망 구조
입력층 2, 은닉층 1, 노드 2, sigmoid 사용
'''

np.random.seed(0)

X = np.array([[1.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [0.0, 0.0]])
y = np.array([[0.0],
              [1.0],
              [1.0],
              [0.0]])
print(f"X: {X}, shape: {np.shape(X)}")
print(f"y: {y}, shape: {np.shape(y)}")

input_size = 2
output_size = 1
hidden_size_list = [2]
activation = 'sigmoid'  # 활성화 함수
weight_init_std = 'sigmoid'

XOR_network = MultiLayerNet(input_size=input_size,
                            hidden_size_list=hidden_size_list,
                            output_size=output_size,
                            activation=activation,
                            weight_init_std=weight_init_std
                            )

print(f"\n네트워크: {XOR_network}")
print(f"파라미터 정보: {XOR_network.params}")
print(f"파라미터 이름: {XOR_network.params.keys()}")

# 가중치와 편향 정보 출력
# print(f"\nW1: {XOR_network.params['W1']}, shape: {np.shape(XOR_network.params['W1'])}")
# print(f"b1: {XOR_network.params['b1']}, shape: {np.shape(XOR_network.params['b1'])}")
# print(f"W2: {XOR_network.params['W2']}, shape: {np.shape(XOR_network.params['W2'])}")
# print(f"b2: {XOR_network.params['b2']}, shape: {np.shape(XOR_network.params['b2'])}")

print(f"\n레이어 정보: {XOR_network.layers}")
print(f"마지막 레이어 정보: {XOR_network.last_layer}")
print("---"*20)

# 가중치와 편향 정보 출력2
# print(f"\nXOR_network.layers['Affine1']W: {XOR_network.layers['Affine1'].W}, shape: {np.shape(XOR_network.layers['Affine1'].W)}")
# print(f"XOR_network.layers['Affine1']b: {XOR_network.layers['Affine1'].b}, shape: {np.shape(XOR_network.layers['Affine1'].b)}")
# print(f"XOR_network.layers['Affine2']W: {XOR_network.layers['Affine2'].W}, shape: {np.shape(XOR_network.layers['Affine2'].W)}")
# print(f"XOR_network.layers['Affine2']b: {XOR_network.layers['Affine2'].b}, shape: {np.shape(XOR_network.layers['Affine2'].b)}")

# 예측
prediction = XOR_network.predict(X) # 모든 전파 수행
print(f"\n예측값 y^out: \n{prediction}, shape: {np.shape(prediction)}")
loss = XOR_network.loss(X, y)
print(f"\n손실 함수의 값: {loss}")

#각 층의 전파 보기
# out = XOR_network.layers['Affine1'].forward(X)
# print(f"\n첫번째 선형변환 y^1: \n{out}, shape: {np.shape(out)}")
# out = XOR_network.layers['Activation_function1'].forward(out)
# print(f"\n비선형변환 z^1: \n{out}, shape: {np.shape(out)}")
# out = XOR_network.layers['Affine2'].forward(out)
# print(f"\n두번째 선형변환 y^2: \n{out}, shape: {np.shape(out)}")
# loss = XOR_network.last_layer.forward(out, y)
# print(f"\n손실값 Loss: \n{loss}, shape: {np.shape(loss)}")

iters_num = 100001
learning_rate = 0.01
for i in range(1, iters_num):
    # 기울기 계산
    grad = XOR_network.gradient(X, y) # 오차역전파 방식(훨씬 빠름)

    # 매개변수 갱신
    for key in XOR_network.params.keys():
        XOR_network.params[key] -= learning_rate * grad[key]

    if i % 20000 == 0:
        print(f"\n{i}번째 기울기: {grad}\n갱신된 파라미터 값: {XOR_network.params}")

        predict = XOR_network.predict(X)
        print(f"\n{i}번째 예측값 y^out: \n{predict}, shape: {np.shape(predict)}")
        # 손실 함수 값 갱신
        loss = XOR_network.loss(X, y)
        print(f"{i}번째 손실 함수 값: {loss}")


print(f"십만번 반복 후 예측값 y^out: \n{np.round(predict, 0)}, shape: {np.shape(predict)}")
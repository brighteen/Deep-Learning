# 손실함수 추가와 학습 과정(for문) 추가

## Loss Function(MeanSquaredError)
$$L = \frac{1}{2}(y_2 - t)^2$$

## 신경망에서 **바이어스(bias)의 역할**과 **기울기(gradient)** 가 어떻게 계산되는지
- `dL/dy2 = dL/db2`, `dL/dy1 = dL/db1`처럼 **출력값의 변화량과 바이어스의 기울기가 같아지는 이유**

![선형변환 두번](<두개의 선형변환레이어.png>)


| 기호     | 의미                                                       |
| ------ | -------------------------------------------------------- |
| $L$    | Loss 함수 (여기서는 MSE)                                       |
| $y_2$  | 두 번째 레이어 출력 (예측값)                                        |
| $t$    | 실제 정답 값                                                  |
| $b_2$  | 두 번째 레이어의 바이어스                                           |
| $dy_2$ | $y_2$에 대한 Loss의 변화율, 즉 $\frac{\partial L}{\partial y_2}$ |
| $db_2$ | $b_2$에 대한 Loss의 변화율, 즉 $\frac{\partial L}{\partial b_2}$ |

### 왜 `dL/dy2 = db2`일까?
이유는 **바이어스**가 출력에 그대로 더해지기 때문

```python
y2 = w2 * y1 + b2
```

Loss를 $L = \frac{1}{2}(y_2 - t)^2$ 라고 할 때,

$$
dy_2 = \frac{\partial L}{\partial y_2} = y_2 - t
$$

$$
\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial y_2} \cdot \frac{\partial y_2}{\partial b_2} = (y_2 - t) \cdot 1 = y_2 - t
$$

여기서 **$\frac{\partial y_2}{\partial b_2} = 1$** 인 이유는, **$b_2$가 출력에 그냥 더해지기 때문**

그래서 결과적으로:

$$
\frac{\partial L}{\partial y_2} = \frac{\partial L}{\partial b_2}
$$

즉, `dy2 = db2`


### 그럼 `dy1 = db1`도 마찬가지일까?

정확히 말하면 **형태는 비슷하지만 원리는 약간 다르다**.

```python
y1 = w1 * x + b1
```

y1이 다음 레이어로 전달되었고, 그 결과를 통해 Loss가 계산됨. 역전파 시, `dy1`은 결국 y1에 대한 Loss의 변화율:

$$
dy_1 = \frac{\partial L}{\partial y_1}
$$

그리고 마찬가지로 바이어스 b1는 y1에 그냥 더해지므로:

$$
\frac{\partial y_1}{\partial b_1} = 1
\Rightarrow \frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial y_1} \cdot 1 = \frac{\partial L}{\partial y_1}
$$

즉, `dy1 = db1`도 성립

- 바이어스는 입력 없이 **그 자체로 더해지는 값**이기 때문에,
- 바이어스의 gradient는 해당 레이어 출력값의 gradient와 **동일**


### 결론
* 바이어스는 출력값에 **직접 더해지는 값**이기 때문에,
* 미분 시 **항상 해당 레이어 출력의 gradient와 같아진다**.

## 딥러닝 모델의 구현과 학습
### 손실 함수 구현

Mean Squared Error(MSE)는 다음과 같이 구현함:
```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2) / y.shape[0]
```

계층으로서의 MSE 구현:
```python
class MeanSquaredError:
    def __init__(self):
        self.y = None    # 예측값
        self.t = None    # 정답값
        self.loss = None # 손실값
        
    def forward(self, x, t):
        self.y = x
        self.t = t
        self.loss = 0.5 * mean_squared_error(self.y, self.t) / self.y.shape[0]
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
```

### 신경망 클래스 구현

```python
class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # 계층 구성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = MeanSquaredError()
```

### 학습 과정 구현

신경망의 학습은 다음 과정으로 진행됨:

1. 미니배치: 훈련 데이터에서 무작위로 일부 데이터를 추출
2. 기울기 계산: 미니배치의 손실 함수 값을 줄이는 방향으로 가중치 매개변수 기울기 계산
3. 매개변수 갱신: 기울기 방향으로 매개변수 값 갱신
4. 반복: 1~3 과정을 반복

```python
# 학습 예시
network = Neural_Network(input_size=2, hidden_size=2, output_size=1)
iters_num = 1000
learning_rate = 0.1

for i in range(iters_num):
    # 기울기 계산
    grad = network.gradient(X_data, y_data)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 출력
    if i % 100 == 0:
        loss = network.loss(X_data, y_data)
        print(f"iteration {i}: loss = {loss}")
```

### XOR 문제에 적용

이진 분류 문제인 XOR 문제에 신경망 적용:

| 입력 | 출력 |
|------|-----|
| [1,1] | 0   |
| [1,0] | 1   |
| [0,1] | 1   |
| [0,0] | 0   |

비선형 활성화 함수(시그모이드)를 사용하여 선형 분리가 불가능한 XOR 문제를 해결할 수 있음.
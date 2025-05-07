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
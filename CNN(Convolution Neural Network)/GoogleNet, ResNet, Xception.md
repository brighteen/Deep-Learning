# GoogLeNet
![](images/GoogleNet_Architecture.png)
- **Inception 모듈**
  - 여러 필터를 병렬로 처리함
  - 다른 DNN 모델보다 파라미터 수가 적고 error율이 낮음 (네트워크 depth가 깊어서임)
  
- **네트워크 구성**
  - 초기 부분: 일반적인 Convolution 연산 사용함 (Inception 모듈 적용 효과가 미미함)
  - 중간 부분: Softmax classifier를 보조 분류기로 2번 사용함
    - Gradient vanishing 발생 여부를 모니터링하기 위함 (성능에 직접적 영향 없음)

---

# ResNet

- **도입 배경**
  - 모델 depth 증가 시 성능 저하(과적합, 기울기 소실 등) 문제 발생함

![](images/깊이에%20따른%20모델%20성능.png)

- **Residual Block**
  - Skip-connection 포함하여 선형, 비선형 변환 및 pooling 과정을 건너뜀
  - 2개의 Convolution 연산마다 적용됨

![](images/ResidualBlock.png)
  
- **기타 특징**
  - 대표적인 모델 사용 예: AlphaFold
  - Feature map 크기 축소 시 pooling 대신 stride=2인 Convolution 연산 사용하여 연산량 조절함
  - 기울기 소실에 대해 강건함

---

# Xception

- **핵심 아이디어**
  - Inception 모듈과 Residual block의 개념 결합함

- **Convolution 연산의 역할**
  - Filter를 통해 cross-channel correlation과 spatial correlation을 동시에 mapping함

- **Inception 모듈의 역할**
  - 두 상관관계를 독립적으로 분석하여 feature 탐지 과정을 쉽고 효율적으로 만듦

- **연산 단계**
  1. 각 channel 별로 spatial feature를 추출함
  2. 추출된 $c$개의 feature map $M_{(i)}$를 concatenation하여 최종 feature map $M$을 구성함
  3. 구성된 feature map $M$에 대해 $(1 \times 1 \times c)$ Convolution 연산을 진행해 channel 간 feature를 추출함

# Transformer - Attention 기반 자연어 처리

## 개요
- **Transformer**의 핵심은 **Attention** 메커니즘
- **Encoder** 구조를 활용하는 대표 모델: **BERT**  
- **Decoder** 구조를 활용하는 대표 모델: **GPT**  

## 3가지 Attention
1. **Self-Attention**  
   - 입력 시퀀스 내 단어들이 서로의 문맥적 관련성을 학습하기 위한 메커니즘임  
2. **Masked Self-Attention**  
   - 주로 **디코더**에서 사용하며, 미래 단어 정보를 참조하지 못하도록 마스킹함  
   - 텍스트 생성 시, 다음 단어 예측을 위해 과거 정보만 보게 함  
3. **Encoder-Decoder Attention**  
   - **인코더**의 출력(입력 문장 정보)과 **디코더**의 상태를 연결함  
   - 번역 등에서 입력 문장의 중요한 단어에 집중하도록 도움  

## 문맥의 중요성
- 단어는 문맥에 따라 여러 의미로 해석 가능함  
- 임베딩 시 **문맥 정보**를 반영해야 함  
- **Self-Attention**으로 문맥 정보를 고려해 단어 벡터를 동적으로 보정함  

## Self-Attention 계산 과정
1. **Attention Score**:  
   $$\alpha_1^{(i)} = q_1^{(i)} \cdot k_j^{(i)}$$ 
   - Query 벡터 $q_1^{(i)}$와 Key 벡터 $k_j^{(i)}$의 **내적**으로 유사도 계산함  
2. **Softmax**:  
   - $\alpha_1^{(i)}$를 Softmax에 통과시켜 확률 분포로 변환함  
3. **Attention Value**:  
   $$a_1^{(i)} = \sum_{j=1}^{n_e} v_j^{(i)} \cdot p_{1,j}^{(i)}$$
   - Value 벡터 $v_j^{(i)}$에 self-attention coefficient $p_{1,j}^{(i)}$를 곱해 가중 평균을 구함  
4. **Attention Vector**:  
   $$\hat{z}_1^{(i)} = \tanh \Bigl(W_1^{(s)(i)} \bigl(q_1^{(i)} + a_1^{(i)}\bigr)\Bigr)$$
   - $q_1^{(i)}$와 $a_1^{(i)}$를 더한 뒤(Concatenate), 매트릭스 $W_1^{(s)(i)}$로 **차원 축소**를 진행하고 $\tanh$ 적용함  
   - $a_1^{(i)}$는 쿼리 벡터 $q_1^{(i)}$를 문맥적으로 **보정**하는 역할을 함  


![](Images\Attention.png)

## 요약
- 입력 벡터 $z_1^{(i)}$를 $W^{(Q)}, W^{(K)}, W^{(V)}$를 통해 Query, Key, Value로 분할함  
- Self-Attention 과정을 통해 문맥을 반영한 $\hat{z}_1^{(i)}$를 생성함  
- Encoder-Decoder 구조에서 Attention 메커니즘은 번역, 질의응답 등 다양한 자연어 처리에서 중요한 역할을 함
import numpy as np
np.random.seed(1)

# w1 = np.random.randn(2,2) # 표준정규분포에서 난수 생성
w1 = np.random.randint(3, size=(2,2)) # 0~1 사이의 정수 난수 생성
# print(f'w1: {w1}')
# print(f'w1.shape: {w1.shape}')

b1 = np.zeros(2)
# print(f'b1: {b1}')
# print(f'b1.shape: {b1.shape}')

# w2 = np.random.randn(2,1)
w2 = np.random.randint(3, size=(2,1)) # 0~1 사이의 정수 난수를 생성
# print(f'w2: {w2}')
# print(f'w2.shape: {w2.shape}')

b2 = np.zeros(1)
# print(f'b2: {b2}')
# print(f'b2.shape: {b2.shape}')

A = {'W1' : w1, 
     'b1' : b1, 
     'W2' : w2, 
     'b2' : b2}

print(f'A: {A}')
print(f'\nW1: {A["W1"]}')
print(f'b1: {A["b1"]}')
print(f'W2: {A["W2"]}')
print(f'b2: {A["b2"]}')

lr = 0.01

A['W1'] = A['W1'] - A['W1'] * lr
A['b1'] = A['b1'] - A['b1'] * lr
A['W2'] = A['W2'] - A['W2'] * lr
A['b2'] = A['b2'] - A['b2'] * lr

print(f'\nA: {A}')
print(f'A의 type: {type(A)}') # dict
print(f'A[W1] : {type(A["W1"])}') # numpy.ndarray
print(f'W1: {A["W1"][0][0]}') # W1: 0.99
print(f'W1: {A["W1"][0,0]}') # W1: 0.99
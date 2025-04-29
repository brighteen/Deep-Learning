import numpy as np

W = np.array([[1, 1], [1, 1]])
print(f'W: {W}')

A = {'W' : W}
print(f'\nW: {A}')

# W 행렬을 딕셔너리로 변환
W_dict = {f"W{i+1}": row.tolist() for i, row in enumerate(W)}
print(f'\nW_dict: {W_dict}')

# 딕셔너리 안에 딕셔너리 포함
nested_dict = {'matrix' : W_dict}
print(f'\nnested_dict: {nested_dict}')
print(f'\nnested_dict.key : {nested_dict.keys()}') # dict_keys(['matrix'])

print('---' * 20)

A['W'] = np.array([[2,2], 
                   [2,2]])

lr = 0.01
A_grad = W # [[1,1], [1,1]]

A['W'] = A['W'] - A_grad * lr # [[2,2], [2,2]] - [[1,1], [1,1]] * 0.01
print(f'\nA: {A}')
print(f'\nA: {A["W"]}')
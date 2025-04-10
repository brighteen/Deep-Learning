import numpy as np

# 3x3 대칭행렬 정의
A = np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]])

print("원본 행렬 A:\n", A)

# 대칭행렬에 최적화된 eigh 함수 사용
eigenvalues, eigenvectors = np.linalg.eigh(A)

print("\n고윳값:")
print(eigenvalues)
print("고윳값의 개수:", len(eigenvalues))

print("\n고유벡터 (각 열이 하나의 고유벡터):")
# print(eigenvectors)
print(np.rint(eigenvectors))
print(type(eigenvectors))

print("\n고유벡터 행렬^T:\n", 
      np.rint(eigenvectors.T))

# 고윳값 분해 재구성으로 A 재현: A = Q Λ Q^T
A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
print("\n재구성한 행렬 A:")
print(np.rint(A_reconstructed))

print("\n재구성한 행렬 A (정수형으로 변환):")
# print(A_reconstructed.astype(int))
# print('round : \n', np.round(A_reconstructed).astype(int))
# print('rint: \n', np.rint(A_reconstructed))

# print(round(0.5))
# print(np.round(0.5))
# print(np.rint(0.5))
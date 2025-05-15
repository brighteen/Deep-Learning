import numpy as np

z = np.array([1,1])
print(f"[z] : {z}, shape: {np.shape(z)}") # 1행 2열
print(f"[z.T] : {z.T}, shape: {np.shape(z.T)}") # 2행 1열
reshape_z1 = np.reshape(z, (1, 2)) # 명시적으로 1행 2열 변환
print(f"[reshape_z1] : {reshape_z1}, shape: {np.shape(reshape_z1)}") # 1행 2열
reshape_z2 = np.reshape(z, (1,-1)) # 
print(f"[reshape_z2] : {reshape_z2}, shape: {np.shape(reshape_z2)}") # 1행 2열

print(reshape_z1 == reshape_z2) # True
if np.array_equal(reshape_z1, reshape_z2): # 
    print("reshape_z1 == reshape_z2")
else:
    print("reshape_z1 != reshape_z2")

c = np.reshape(z, (2,1))
print(f"[c] : {c}, shape: {np.shape(c)}") # 2행 1열
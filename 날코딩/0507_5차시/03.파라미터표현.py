import numpy as np

params = {}

params['W1'] = np.random.rand(2,2)
# params['b1'] = np.random.rand(2,1)
params['b1'] = np.zeros(2)

print(f"params: \n{params}")
print(f"\nparams type: {type(params)}")
print(f"\nW1 shape: {params['W1'].shape}")
print(f"W1: \n{params['W1']}")

print(f"\nb1 shape:{params['b1'].shape}")
print(f"b1: \n{params['b1']}")


print('\n---')
x = np.array([[1,2,3,4], [5,6,7,8]])
print(f"x: \n{x}\n")
print(f"x shape: {x.shape}") # (2,4)
print(f"x shape[0]: {x.shape[0]}") # 2
print(f"x shape[1]: {x.shape[1]}\n") # 4
print(f"x reshape 4,2: \n{x.reshape(4,2)}")
print(f"\nx.T: \n{x.T}")

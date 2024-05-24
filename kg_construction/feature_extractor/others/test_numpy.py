import numpy as np

x = np.array([
	[1, 2, 3],
	[1, 4, 9],
	[1, 8, 27]
	])

print(x)

x = x.reshape(x.shape+(1,))
x = np.concatenate([x, x, x], axis=2)
print(x.shape)
print(x)
print(x[:,:,1])

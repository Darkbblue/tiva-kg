import h5py
import numpy as np

path = '/m/en/cat/audio/0'
path2 = '/m/en/cat/visual/0'
pathr = '/m/en/r,@m@en@cat,@m@en@dog/image/0'
with h5py.File('h5.h5', 'w') as f:
	#f.create_dataset(path, data=np.array([1, 2, 3]))
	f[path] = np.array([1, 2, 3])
	f.create_dataset(path2, data=np.array([3, 2, 1]))
	f.create_dataset(pathr, data=np.zeros((2, 2)))

with h5py.File('h5.h5', 'r') as f:
	print(f[path])
	print(f[path][:])
	print(f[path2][:])
	print(list(f['/m/en'].keys()))

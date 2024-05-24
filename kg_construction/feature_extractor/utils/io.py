import os
import json
import h5py

TEST = False

# ----- init ----- #

size = None
rank = None
h5_path = None
fh5 = None

with open('mm_list.json') as f:
	mm_list = json.load(f)

def init(g_size, g_rank):
	global size
	global rank
	global h5_path
	global fh5

	size = g_size
	rank = g_rank

	if TEST:
		h5_path = 'test.hdf5'
	else:
		h5_path = 'features.hdf5'
	h5_path = str(rank) + h5_path
	fh5 = h5py.File(h5_path, 'a')


def load(modal):
	if TEST:
		ret = mm_list[modal][:1000]
	else:
		ret = mm_list[modal]

	stride = int(len(ret) / size)
	if rank == size - 1:
		return ret[rank*stride:]
	else:
		return ret[rank*stride:rank*stride+stride]


def save(node):
	if 'result' in node:
		di = node['relative_path'] + '/feature'
		if di in fh5:
			del fh5[di]
		#fh5[di] = node['result']
		fh5.create_dataset(di, data=node['result'], compression='gzip')
		del node['result']

	if 'frame' in node:
		di = node['relative_path'] + '/frame'
		if di in fh5:
			print('repeat frame')
			del fh5[di]
		#fh5[di] = node['frame']
		fh5.create_dataset(di, data=node['frame'], compression='gzip', compression_opts=9)
		del node['frame']

	if 'motion' in node:
		di = node['relative_path'] + '/motion'
		if di in fh5:
			print('repeat motion')
			del fh5[di]
		#fh5[di] = node['motion'].detach().cpu().numpy()
		fh5.create_dataset(di, data=node['motion'].detach().cpu().numpy(), compression='gzip', compression_opts=9)
		del node['motion']


def show_h5(node):
	pass


def milestone():
	global fh5
	fh5.close()
	fh5 = h5py.File(h5_path, 'a')


def end():
	fh5.close()

import json
import h5py

TEST = False

with open('mm_list.json') as f:
	mm_list = json.load(f)

def load(modal, rank, size):
	if TEST:
		ret = mm_list[modal][:10000]
	else:
		ret = mm_list[modal]

	stride = int(len(ret) / size)
	if rank == size - 1:
		return ret[rank*stride:]
	else:
		return ret[rank*stride:rank*stride+stride]


fh5 = h5py.File('all.hdf5', 'w')
for rank in range(8):
	s = h5py.File(str(rank)+'features.hdf5', 'r')

	try:
		with open(str(rank)+'fail_list.json', 'r') as f:
			fail_list = json.load(f)
	except:
		fail_list = []
	fail = []
	for e in fail_list:
		fail.append(e['relative_path'])

	for modal in ['image', 'audio', 'visual']:
		target_list = load(modal, rank, 8)
		for i, e in enumerate(target_list):
			if i % int(len(target_list)/20) == 0:
				print(i, '/', len(target_list), 'rank', rank, modal)

			if e['relative_path'] in fail:
				continue
			for t in ['feature', 'motion', 'frame']:
				di = e['relative_path'] + '/' + t
				if (di in s) and (di not in fh5):
					fh5.create_dataset(di, data=s[di], compression='gzip', compression_opts=9)

	s.close()


fh5.close()

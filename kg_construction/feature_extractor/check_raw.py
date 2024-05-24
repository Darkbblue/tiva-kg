# 检查h5文件是否内容齐全

import os
import json
import h5py


# 初始化

with open('../mmkg_data_v3/entities.json', 'r') as f:
	entities_all = json.load(f)
with open('../mmkg_data_v3/relations.json', 'r') as f:
	relations_all = json.load(f)

f = h5py.File('all.hdf5', 'r')

total = 0
detail = {
	'audio': 0,
	'image': 0,
	'visual': 0
}


# 检查存在性的方法

def check_mm(prop):
	detected = False
	for T in ['audio', 'image', 'visual']:
		if T in prop:
			for i in prop[T]:
				detected = i['@id'] not in f
				if not detected:
					if T == 'visual':
						detected = i['@id']+'/motion' not in f or i['@id']+'/frame' not in f
					else:
						detected = i['@id']+'/feature' not in f

	return detected


# 外循环

len_e = len(entities_all)
for i, e in enumerate(entities_all.values()):
	if check_mm(e):
		total += 1
	if i % 500 == 0:
		print('entity %d / %d | detected %d' % (i, len_e, total))

len_r = len(relations_all)
for i, r in enumerate(relations_all.values()):
	if 'properties' not in r:
		continue
	if check_mm(r['properties']):
		total += 1
	if i % 5000 == 0:
		print('relation %d / %d | detected %d' % (i, len_r, total))


# 结果

print('total invalid dir: %d' % total)
print('audio: %d\nimage: %d\nvisual: %d' % (detail['audio'], detail['image'], detail['visual']))

f.close()

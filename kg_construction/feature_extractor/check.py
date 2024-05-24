import json
import h5py
from utils import io

io.init(1, 0)

visual_list = io.load('visual')
print(len(visual_list))
visual = []
for e in visual_list:
	visual.append(e['relative_path'])
#print(visual)

fail = []
for i in range(8):
	try:
		with open(str(i)+'fail_list.json', 'r') as f:
			fail_list = json.load(f)
	except:
		fail_list = []
	for e in fail_list:
		fail.append(e['relative_path'])
print(fail)

f = h5py.File('all.hdf5', 'r')


# 确认列表中的项目都在文件中

for e in visual_list:
	if e['relative_path'] in fail:
		continue
	assert e['relative_path'] in f


# 检查数据内容
sam = visual_list[26]
print(list(f[sam['relative_path']]))
sam = f[sam['relative_path']]
print(sam['motion'])
print(sam['frame'])


# 确认文件中的内容没有冗余

path = '/m'
last_inner = ''
while len(path) > 0:
	#print(path)

	inners = list(f[path])
	inwards = True
	next_inner = inners[0]

	if last_inner in inners:
		if last_inner == inners[-1]:
			inwards = False
		else:
			for i, e in enumerate(inners):
				if last_inner == e:
					next_inner = inners[i+1]

	if 'frame' in inners and 'motion' in inners and len(inners) == 2:
		inwards = False
		assert path in visual

	if inwards:
		path = path + '/' + next_inner
	else:
		tail = path.split('/')[-1]
		last_inner = tail
		path = path[:len(path)-len(tail)-1]


f.close()

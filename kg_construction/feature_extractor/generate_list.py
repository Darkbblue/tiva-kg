# 用于完成对多模态数据的抽样，结果记录为 list

import json


# 初始化

with open('../mmkg_data_v2/entities.json', 'r') as f:
	entities_all = json.load(f)
with open('../mmkg_data_v2/relations.json', 'r') as f:
	relations_all = json.load(f)

image_list = []
visual_list = []
audio_liet = []


# 取用多模态数据

def sample_mm(prop, text_base):
	for T, target in [('audio', audio_liet), ('image', image_list), ('visual', visual_list)]:
		if T in prop:
			for i in prop[T]:
				target.append({
					'path': '/DATA/DATANAS1/kg' + i['@id'] + '.' + i['form'],
					'relative_path': i['@id'],
					'form': i['form']
					})


# 外循环

for i, e in enumerate(entities_all.values()):
	sample_mm(e, e)

for i, r in enumerate(relations_all.values()):
	if 'properties' not in r:
		continue
	sample_mm(r['properties'], r)

print(len(image_list), len(visual_list), len(audio_liet))

with open('mm_list.json', 'w') as f:
	f.write(json.dumps({
		'image': image_list,
		'visual': visual_list,
		'audio': audio_liet
		}))

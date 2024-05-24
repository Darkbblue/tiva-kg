'''
查看当前工作进度
'''

import json

with open('history/to_fetch.json', 'r') as f:
	print('to_fetch', len(json.load(f)))
with open('history/to_extend.json', 'r') as f:
	print('to_extend', len(json.load(f)))
with open('history/finished.json', 'r') as f:
	print('finished', len(json.load(f)))

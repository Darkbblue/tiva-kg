''' 多模态扩展相关方法 '''

import re
import os
from .uri import uri_to_label
from . import fetch, download_sounds, download_images, download_gifs


def multimodeL_path(uri, model):
	''' 传入节点或边的 uri，并指定模态类型，转化为相应多模态信息的可用路径，路径后加上 '''
	path = 'm' + fetch.filename_transform(uri)[1 : ] # 开头 uri 类型修改
	path = re.findall('(.*)\..*?', path)[0] # 去掉结尾文件类型
	path = path + '/' + model # 增加一级模态类型相关的目录
	return path


def multimodel_uri(uri, model):
	''' 传入节点或边的 uri，转化为多模态信息的基础 uri，需要手动拼接上序号方可使用 (e.g. +'1') '''
	return '/m' + uri[2 : ] + '/' + model + '/'


def get_text(uri):
	''' 根据 uri 得到可以用于搜索的文本 '''
	if uri[1] == 'c': # 点
		text = uri_to_label(uri)
	else: # 边
		blocks = re.findall('^.*\[(.*?)\]', uri)[0].split(',')
		start_node = uri_to_label(blocks[1]) + ' '
		end_node = uri_to_label(blocks[2])

		# 根据具体的关系类型进行文本转换
		# 根据具体的边类型提供不同的方案，注意如果是空的就不要加空格，否则要加空格
		edge = blocks[0]
		if edge in ['/r/PartOf/', '/r/HasProperty/', '/r/AtLocation/', '/r/Causes/']:
			relation = ''
		elif edge == '/r/HasA/':
			relation = 'has '
		elif edge == '/r/UsedFor/':
			relation = 'used for '
		elif edge == '/r/CapableOf/':
			relation = 'can '
		elif edge == '/r/MadeOf/':
			relation = 'is made of '
		elif edge == '/r/ReceivesAction/':
			relation = 'can be '
		elif edge == '/r/CreatedBy/':
			relation = 'created by '
		elif edge == '/r/LocatedNear/':
			relation = 'near '

		text = (start_node + relation + end_node).replace(' ', '+')
	return text


def add_multimodel_node_audio(text : str, path : str):
	'''
	传入的参数中，text 为直接用于搜索的字符串，path 为用于保存文件的路径
	path + '/0.wav' 进行类似左侧的字符串拼接后，可以得到文件名，注意编号从 0 开始
	path 可直接用于 os.makedirs(path)，拼接后的文件名可用于 with open(filename, 'w') as f
	期待返回一个列表，每一项都是所保存的文件的格式，例如 ['wav', 'wav', 'mp3']
	'''
	#print('path', path)
	#print('filename', path+'/1.wav')
	if not os.path.exists(path): # 如果目录不存在，则补上目录
		os.makedirs(path)
	return download_sounds.download_sound(text, path)


def add_multimodel_node_image(text : str, path : str):
	if not os.path.exists(path): # 如果目录不存在，则补上目录
		os.makedirs(path)
	return download_images.download_image(text, path)


def add_multimodel_node_visual(text : str, path : str):
	if not os.path.exists(path): # 如果目录不存在，则补上目录
		os.makedirs(path)
	return download_gifs.download_gif(text, path)


def add_multimodel(page, model_type):
	''' 向页面中添加多模态信息 '''
	uri = page['@id']
	model_info = {} # 页面中新增的记录多模态信息的域，暂时用一个单独的词典记录
	text = get_text(uri)

	# 音频
	newly_added = add_multimodel_node_audio(text, multimodeL_path(uri, 'audio')) # 下载信息并返回新增项的列表
	#print(newly_added)
	if len(newly_added) > 0:
		model_info['audio'] = [] # 列表初始化为空
		for i, form in enumerate(newly_added):
			model_info['audio'].append({'@id' : multimodel_uri(uri, 'audio')+str(i), 'form' : form})
		#print('\t\taudio added')

	# 图片
	newly_added = add_multimodel_node_image(text, multimodeL_path(uri, 'image')) # 下载信息并返回新增项的列表
	#print(newly_added)
	if len(newly_added) > 0:
		model_info['image'] = [] # 列表初始化为空
		for i, form in enumerate(newly_added):
			model_info['image'].append({'@id' : multimodel_uri(uri, 'image')+str(i), 'form' : form})
		#print('\t\timage added')

	# 视觉
	newly_added = add_multimodel_node_visual(text, multimodeL_path(uri, 'visual')) # 下载信息并返回新增项的列表
	#print(newly_added)
	if len(newly_added) > 0:
		model_info['visual'] = [] # 列表初始化为空
		for i, info in enumerate(newly_added):
			model_info['visual'].append({'@id' : multimodel_uri(uri, 'visual')+str(i), 'with-audio' : info[0], 'form' : info[1]})
		#print('\t\tvisual added')

	page['multi-model'] = model_info # 加入到原文本中
	return page

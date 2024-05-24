'''
从 ConceptNet 获取点和边的文本信息
'''


# 请求数据 -> 首先在本地查找 -> 本地查找失败后下载 -> 下载完成后返回，由外层决定何时保存

# 点的分页机制需要特殊处理
# 服务器后端的存储肯定是一整页的，根据传入的参数对 edge 进行分片返回，所以下载时应该合并为整页

# 边的 uri 比较复杂，可能需要进行路径替换；格式为/a/[关系,起点,终点]，需要对中括号内的斜杠进行替换
# 由 uri，需要能够得到目录结构，以及能够直接访问到正确目标文件的路径，前者对边和点是通用的
# 后者点只需要去掉开头斜杠，而边需要额外的字符替换
# 点最后的 /n/，这种都是另一个点，和不加 /n/ 的点不一样，所以不再额外处理

# 对节点的边列表进行过滤
# 既要有判断过滤条件的方法，又要能够删除不符合要求的边

import os
import re
import json
import requests
import subprocess
from time import sleep
from . import proxy

def download(uri):
	''' 下载所给 uri 指定的页面，返回词典格式 '''
	headers = {
		#'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0',
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
		'Connection': 'close'
	}
	proxy_ip = proxy.get_proxy()
	proxies = {'http': 'http://{}'.format(proxy_ip), 'https': 'https://{}'.format(proxy_ip)}

	i = 0 # 重试计数器
	while i < 10:
		try:
			response = requests.get('http://api.conceptnet.io'+uri, headers=headers, proxies=proxies, timeout=5)
			if response.status_code == 200: # 若正常获取则返回结果
				return response.json()
			else:
				raise requests.exceptions.RequestException
		except requests.exceptions.RequestException:
			print('!reconnecting!')
			sleep(5)
			i += 1 # 若出现错误，则进行有限次数重试
	raise requests.exceptions.RequestException


def filename_transform(uri):
	''' 将 uri 转化为完整的文件路径 '''
	element_type = uri[1] # 'a' for edge and 'c' for node
	uri = uri[1 : ] # 去掉开头的斜杠
	if element_type == 'a': # 若是边，则需要对中括号内的斜杠进行字符替换
		uri = re.findall('^(.*)\[.*?\]', uri)[0] + re.findall('^.*(\[.*?\])', uri)[0].replace('/', '@')
		uri += '.json'
	elif element_type == 'c': # 若是点，只需要加上扩展名
		uri += '.json'
	else: # 若是多模态信息
		if re.match('^.*\[.*?\].*', uri):
			blocks = re.findall('^(.*)(\[.*?\])(.*)', uri)[0]
			uri = blocks[0] + blocks[1].replace('/', '@') + blocks[2]
	return uri


def path_transform(uri):
	''' 将 uri 转化为目录结构 '''
	element_type = uri[1] # 'a' for edge and 'c' for node
	if element_type == 'c': # 若是点
		path = re.findall('^/(.*)/.*?', uri)[0] # 去掉最后一级的文件名即可
	else: # 若是边
		path = re.findall('^/(.*)/\[.*?\]', uri)[0] # 最后一级文件名包含在括号内
	return path


def get_page(uri):
	''' 获取指定页面，解析为词典格式 '''
	need_to_save = False
	filename = filename_transform(uri)
	# 首先尝试从本地获取
	try:
		with open(filename, 'r') as f:
			page = json.loads(f.read())
	# 若本地没有，再尝试下载
	except:
		page = download(uri) # 获得点的数据
		need_to_save = True

	return page, need_to_save


def get_node(uri):
	''' 获取 uri 指定的点，需要处理分页展示的问题，合并为一整页 '''
	node, is_from_web = get_page(uri)
	if is_from_web and 'view' in node: # 若是分页提供的点，则合并为一整页
		#print('downloading paginated node')
		page = node
		while 'view' in page and 'nextPage' in page['view']:
			page = download(page['view']['nextPage'])
			node['edges'] += page['edges']
			#if (len(node['edges']) % 500 == 0):
				#print(len(node['edges']))
		del node['view']
	return node, is_from_web


def get_edge(uri):
	''' 获取 uri 指定的边 '''
	return get_page(uri)


def get_multi_model(info):
	''' 直接将一个包含 id, 格式等信息的词典传入 '''
	filename = filename_transform(info['@id'])
	filename = filename + '.' + info['form']
	with open(filename, 'rb') as f:
		content = f.read()
	return content


def save_to_local(uri, content):
	''' 将json格式的数据保存到本地 '''
	path = path_transform(uri) # 目录结构，去除了最后一级的文件名
	filename = filename_transform(uri) # 包含文件名的完整路径
	if not os.path.exists(path): # 如果目录不存在，则补上目录
		os.makedirs(path)
	with open(filename, 'w') as f: # 写入文件
		f.write(json.dumps(content))


def can_accept(element):
	''' 根据传入的词典格式的 element，判断是否可以接受，可以接受的参数类型包括单独的点、单独的边、点的边列表中的边 '''
	element_type = element['@id'][1] # 'a' for edge and 'c' for node
	if element_type == 'c': # 若是点
		if re.findall('^/.*?/(.*?)/.*', element['@id'])[0] != 'en': # 语言非英语
			return False

	elif element_type == 'a': # 若是边
		if element['end']['@id'][1] != 'c' or element['start']['@id'][1] != 'c': # 两端点包含点和关系外的类型的元素
			return False
		if element['end']['language'] != 'en' or element['start']['language'] != 'en': # 两端点含非英语
			return False
		if element['rel']['@id'] not in [
			'/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/HasProperty', '/r/MannerOf',
			'/r/MadeOf', '/r/ReceivesAction', '/r/AtLocation', '/r/Causes', '/r/HasSubevent', '/r/HasFirstSubevent',
			'/r/HasLastSubevent', '/r/HasPrerequisite', '/r/MotivatedByGoal', '/r/ObstructedBy', '/r/Desires',
			'/r/CreatedBy', '/r/DistinctFrom', '/r/SymbolOf', '/r/DefinedAs', '/r/LocatedNear', '/r/HasContext',
			'/r/SimilarTo', '/r/CausesDesire'
		]:
			return False
		if element['weight'] < 1.0: # 若权重过小
			return False

	else: # 其他类型，不保留
		return False

	return True # 通过全部判定


def need_extend(uri):
	''' 传入 uri，返回是否需要进行多模态扩展 '''
	if uri[1] == 'c': # 点
		return True
	else:
		blocks = re.findall('^.*\[(.*?)\]', uri)[0].split(',')
		relation = blocks[0]
		if relation in ['/r/IsA/', '/r/MannerOf/', '/r/HasSubevent/', '/r/HasFirstSubevent/', '/r/HasLastSubevent/',
			'/r/HasPrerequisite/', '/r/MotivatedByGoal/', '/r/ObstructedBy/', '/r/Desires/', '/r/DistinctFrom/',
			'/r/SymbolOf/', '/r/DefinedAs/', '/r/HasContext/', '/r/SimilarTo/', '/r/CausesDesire/', '/r/NotDesires/']:
			return False
		else:
			return True


def filter_node(node):
	''' 过滤掉所给节点关联边中不合适者 '''
	node['edges'] = [edge for edge in node['edges'] if can_accept(edge)]
	return node


def delete_uri(uri):
	''' 删除指定 uri 对应的文件 '''
	filename = filename_transform(uri)
	cmd = 'rm -f {}'.format(filename.replace("'", "\\'"))
	subprocess.run(cmd, shell=True)

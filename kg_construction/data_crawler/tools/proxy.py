''' 提供代理 '''

import random


def get_proxy():
	proxy_list = [
		'23.254.82.177:1337',
		'45.57.159.187:1337',
		'45.57.161.133:1337',
		#'107.152.233.145:1337',
		'107.152.233.116:1337'
		]
	return 'tartessossian:O0TEVmy@' + random.choice(proxy_list)


def get_proxy_list():
	proxy_list_origin = [
		'23.254.82.177:1337',
		'45.57.159.187:1337',
		'45.57.161.133:1337',
		'107.152.233.145:1337',
		'107.152.233.116:1337'
		]
	proxy_list = []
	for proxy in proxy_list_origin:
		proxy_list.append('tartessossian:O0TEVmy@' + proxy)
	return proxy_list

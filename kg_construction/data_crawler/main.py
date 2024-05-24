import subprocess
import threading
import datetime
import time
import json
from tools import fetch, extend, transport, add
from py2neo import Graph


# 以下参数影响并发度
fetch_limit = 8
extend_limit = 5
# 以下是其他参数
root = '/c/en/pet'


to_fetch = {}
on_fetch = {}
to_extend = {}
on_extend = {}
finished = {}

lock_to_fetch = threading.Lock()
lock_to_fetch_monitor = threading.Lock()  # 专用于保存时阻塞分发过程的锁
lock_on_fetch = threading.Lock()
lock_to_extend = threading.Lock()
lock_on_extend = threading.Lock()
lock_finished = threading.Lock()
lock_add = threading.Lock()

graph = Graph('http://localhost:7474', username='neo4j', password='root')


# ----- monitors ----- #


def monitor_fetch():
	while True:
		# 在存在空闲时创建新的线程
		while len(on_fetch) < fetch_limit:
			# 若暂无待拉取页面则跳过本轮
			if len(to_fetch) <= 0:
				break

			with lock_to_fetch_monitor:
				with lock_to_fetch:
					uri = list(to_fetch.keys())[0]
					del to_fetch[uri]
			with lock_on_fetch:
				task = threading.Thread(target=worker_fetch, args=(uri,))
				task.start()
				on_fetch[uri] = task

		# 终止条件
		# 由于 to_fetch 只可能由 on_fetch 中的任务添加
		# 所以若两者均为空，则可以终止
		if len(to_fetch) <= 0 and len(on_fetch) <= 0:
			time.sleep(10)
			if len(to_fetch) <= 0 and len(on_fetch) <= 0:
				break

		# 等待当前任务完成或有新的任务被发现
		time.sleep(5)


def monitor_extend():
	while True:
		# 在存在空闲时创建新的线程
		while len(on_extend) < extend_limit:
			# 若暂无待拉取页面则跳过本轮
			if len(to_extend) <= 0:
				break

			with lock_to_extend:
				uri = list(to_extend.keys())[0]
				del to_extend[uri]
			with lock_on_extend:
				task = threading.Thread(target=worker_extend, args=(uri,))
				task.start()
				on_extend[uri] = task

		# 终止条件
		if len(to_fetch) + len(on_fetch) + len(to_extend) + len(on_extend) <= 0:
			time.sleep(30)
			if len(to_fetch) + len(on_fetch) + len(to_extend) + len(on_extend) <= 0:
				break

		# 等待当前任务完成或有新的任务被发现
		time.sleep(10)


# ----- workers ----- #


def worker_fetch(uri):
	# 拉取任务
	try:
		if uri[1] == 'c': # 点
			print(uri)
			# 获得页面
			node, newly_downloaded = fetch.get_node(uri)
			need_extend = True
			# 对新下载的点进行过滤与保存，若已经在本地则跳过
			if newly_downloaded:
				node = fetch.filter_node(node)
				fetch.save_to_local(uri, node)
			# 获取新增边
			new_fetch = [ edge['@id'] for edge in node['edges'] ]

		else: # 边
			print('\t', uri)
			# 获得页面
			edge, newly_downloaded = fetch.get_edge(uri)
			# 只在需要扩展时临时存储
			need_extend = fetch.need_extend(uri)
			if need_extend:
				fetch.save_to_local(uri, edge)
			# 获取新增点
			new_fetch = [ node['@id'] for node in [edge['start'], edge['end']] if fetch.can_accept(node) ]

		# 新增页面
		with lock_to_fetch:
			for page in new_fetch:
				if page not in finished and page not in on_extend and page not in to_extend and page not in on_fetch:
					to_fetch[page] = True

		# 当前 uri 移入待扩展池或者完成池
		with lock_on_fetch:
			del on_fetch[uri]

		# 若需要进行多模态扩展
		if need_extend:
			with lock_to_extend:
				to_extend[uri] = True
		# 若不进行多模态扩展
		else:
			new_element = node if uri[1] == 'c' else edge
			worker_add(uri, new_element)
			with lock_finished:
				finished[uri] = True

	except Exception as e:
		# 放弃任务，将当前任务放回池子
		print('fetch failed', uri)
		print(e)
		with lock_on_fetch:
			if uri in on_fetch:
				del on_fetch[uri]
		with lock_to_fetch:
			to_fetch[uri] = True


def worker_extend(uri):
	try:
		# 拉取任务
		print('\t\t', uri)
		if uri[1] == 'c': # 点
			node = fetch.get_node(uri)[0] # 获得当前点
			extend.add_multimodel(node, 'c') # 添加多模态信息
			fetch.save_to_local(uri, node)
			worker_add(uri, node)

		else: # 边
			edge = fetch.get_edge(uri)[0] # 获得当前边
			extend.add_multimodel(edge, 'a') # 添加多模态信息
			fetch.save_to_local(uri, edge)
			worker_add(uri, edge)

		# 删除本地文本文件
		fetch.delete_uri(uri)

		# 当前 uri 移入完成池
		with lock_on_extend:
			del on_extend[uri]
		with lock_finished:
			finished[uri] = True
		print('\t\tfinished ', uri)

	except Exception as e:
		# 放弃任务，将当前任务放回池子
		print('extend failed', uri)
		print(e)
		with lock_on_extend:
			if uri in on_extend:
				del on_extend[uri]
		with lock_to_extend:
			to_extend[uri] = True


def worker_add(uri, info):
	'''
	将拉取好的点或边数据加入数据库
	应当保证只对加完多模态信息的点/边调用，或者是对确定不加多模态信息的点/边调用
	'''
	with lock_add:
		if uri[1] == 'c': # 点
			add.add_entity(info, graph)
		else: # 边
			add.add_relation(info, graph)


# ----- assists ----- #


def assist_save():
	''' 对任务进度进行保存 '''
	# 阻塞 monitor_fetch，并等待 fetch 任务全部完成
	with lock_to_fetch_monitor:
		while len(on_fetch) > 0:
			time.sleep(5)

		# 阻塞 monitor_extend，由于 fetch 任务已经全部完成所以不会死锁
		# 并等待 extend 任务全部完成
		with lock_to_extend:
			while len(on_extend) > 0:
				time.sleep(5)

			# 保存三个相对静态的池
			with open('history/to_fetch.json', 'w') as f:
				f.write(json.dumps(to_fetch))
			with open('history/to_extend.json', 'w') as f:
				f.write(json.dumps(to_extend))
			with open('history/finished.json', 'w') as f:
				f.write(json.dumps(finished))
			print('successfully saved snapshot')
			log = 'to_fetch_'+str(len(to_fetch))+'_to_extend_'+str(len(to_extend))+'_finished_'+str(len(finished))

	print(log)
	# mail
	mail_theme = '每小时爬虫进度提醒'
	mail_content = log
	transport.email(mail_theme, mail_content)


def assist_transport(name):
	''' 用于将本地的多模态数据传输到远端 '''
	try:
		# 阻塞 monitor_extend，并等待 extend 任务全部完成
		with lock_to_extend:
			while len(on_extend) > 0:
				time.sleep(5)
			# 执行重命名
			transport.rename(name)

		# 执行剩余的任务，已经不再需要阻塞
		transport.deliver_content(name)
	except Exception as e:
		print(e)
		mail_theme = '传输失败_%s' % name
		mail_content = str(e)
	else:
		mail_theme = '传输成功_%s' % name
		mail_content = '请及时执行解压缩'
	transport.email(mail_theme, mail_content)


def assist_clock():
	''' 利用系统时间对两个辅助线程进行管理 '''
	while True:
		# 等待到下一个整点
		curTime = datetime.datetime.now()
		desTime = curTime.replace(minute=0, second=0, microsecond=0)
		desTime = desTime + datetime.timedelta(hours=1)
		time.sleep((desTime - curTime).total_seconds())

		# 执行保存
		assist_save()
		# 执行传输，设置为非守护线程，确保传输过程不被打断
		if desTime.hour in [0, 3, 6, 9, 12, 15, 18, 21]:
			args = (str(desTime).replace(' ', '_').replace(':', '-'),)
			threading.Thread(target=assist_transport, args=args, daemon=False).start()


def main():
	task_fetch = threading.Thread(target=monitor_fetch)
	task_extend = threading.Thread(target=monitor_extend)
	task_clock = threading.Thread(target=assist_clock, daemon=True)

	task_fetch.start()
	task_extend.start()
	task_clock.start()

	task_fetch.join()
	task_extend.join()

	assist_save()


if __name__ == '__main__':
	try:
		with open('history/to_fetch.json', 'r') as f:
			to_fetch = json.load(f)
		with open('history/to_extend.json', 'r') as f:
			to_extend = json.load(f)
		with open('history/finished.json', 'r') as f:
			finished = json.load(f)
	except:
		to_fetch[root] = True

	main()

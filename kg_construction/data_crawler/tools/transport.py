'''
将存储在本地的多模态数据传输到远端

使用方法：先阻塞地调用 rename() 将当前多模态文件夹重命名，然后调用deliver_content()，后者可以和主要爬取流程并发执行
下面是写法，你也可以在 main.py 中的 assist_clock() 看到类似的写法

name = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
rename(name)  # 这里建议阻塞住
deliver_content(name)  # 这里可以并行

注：建议使用 datetime 来生成文件名

使用前记得修改 transport() 中的 path_server 和 server，这两者执行传输目标节点以及在其中的存储路径
'''

import subprocess


def rename(name):
	''' 对文件进行重命名 '''
	cmd = 'mv m {}'.format(name)
	ret = subprocess.run(cmd, shell=True)
	if ret.returncode != 0:
		print('subprocess error')
		raise Exception('rename error')


def compress(name):
	''' 对指定文件进行分卷打包压缩，然后删除原文件 '''
	cmd = 'tar -cvzf - ./{} | split -d -b 500m - {}'.format(name, name+'.tar')
	ret = subprocess.run(cmd, shell=True)
	if ret.returncode != 0:
		print('subprocess error')
		raise Exception('compress error')

	cmd = 'rm -rf {}'.format(name)
	ret = subprocess.run(cmd, shell=True)
	if ret.returncode != 0:
		print('subprocess error')
		raise Exception('compress error at removing section')


def transport(name):
	''' 将指定文件组发送到服务器上 '''
	path_local = ''
	path_server = '/DATA/DATANAS1/kg/raw/'
	server = 'bhuang@166.111.5.203'
	cmd = 'scp -P 2333 {}{}.tar* {}:{}'.format(path_local, name, server, path_server)
	ret = subprocess.run(cmd, shell=True)
	if ret.returncode != 0:
		print('subprocess error')
		raise Exception('transport error')


def clear(name):
	cmd = 'rm -rf {}*'.format(name)
	print(cmd)
	ret = subprocess.run(cmd, shell=True)
	if ret.returncode != 0:
		print('subprocess error')
		raise Exception('clear error')


def deliver_content(name):
	''' 执行打包、传输、清理步骤，重命名步骤应单独调用 '''
	compress(name)
	transport(name)
	clear(name)


def email(title, content):
	''' 发送邮件，邮件目标地址直接在本函数中修改 '''
	# 下面的邮件目标地址列表可修改
	mail_receive_addr = [
		#'mby18@mails.tsinghua.edu.cn',
		#'1351800490@qq.com',
		'darkbblue@outlook.com',
	]

	for addr in mail_receive_addr:
		cmd = 'echo %s | s-nail  -s  %s  %s' % (content, title, addr)
		ret = subprocess.run(cmd, shell=True)
		if ret.returncode != 0:
			print('mail subprocess error')

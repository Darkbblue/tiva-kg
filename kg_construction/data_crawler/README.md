- scripts: 在多模态数据接收节点上运行的脚本，会自动化地将分卷压缩包拼接起来、解压缩、数据移动到m文件夹下，然后删除压缩包；注意使用前应保证当前目录下存在一个名称为'm'的子文件夹

- tools: 爬虫组件
	- add.py: 向数据库中添加记录
	- download_gifs.py: 下载动图
	- download_images.py: 下载图片
	- download_sounds: 下载音频
	- extend.py: 进行多模态数据扩展
	- fetch.py: 从ConceptNet中爬取文本数据
	- proxy.py: 提供代理
	- transport.py: 将多模态数据打包传输到本地节点
	- uri.py: 官方提供的工具，目前只用到将uri转换成label的功能

- clear.py: 将数据库中的数据全部清除
- history.py: 查看历史记录的长度，用来检查爬取进度

- **main.py**: 主函数，核心控制流位于此处

- test.py: 杂七杂八的测试
- test_proxy.py: 测试代理是否可用
- test_transport.py: 测试多模态打包传输功能


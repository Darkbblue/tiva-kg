'''
清理现场，debug 使用
'''

from py2neo import Graph
graph = Graph('http://localhost:7474', username='neo4j', password='root')
graph.delete_all()

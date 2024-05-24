import os
import sys
from tools import transport


if len(sys.argv) == 2:
	name = str(sys.argv[1])
	if not os.path.exists(name+'.tar00'):
		if not os.path.exists(name):
			transport.rename(name)
		transport.compress(name)
	transport.transport(name)
	transport.clear(name)
else:
	print('usage: python start_transport.py <name>')

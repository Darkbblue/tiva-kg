size = 3
a = list(range(10))

stride = int(len(a) / size)
for rank in range(size):
	if rank == size - 1:
		print(a[rank*stride:])
	else:
		print(a[rank*stride:rank*stride+stride])

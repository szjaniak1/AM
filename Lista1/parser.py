def parse(path: str):
	file = open(str(sys.argv[1]), 'r')
	dim, dim_num = file.readline().split(':')
	num = int(dim_num)

	result = 
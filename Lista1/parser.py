def parse(path: str):
	file = open(path)
	dim, dim_num = file.readline().split(':')
	size = int(dim_num) + 1
	result = [[0 for i in range(size)] for j in range(size)]
	for line in file:
		v1, v2, weight = line.split(' ')
		result[int(v1)][int(v2)] = int(weight)
		result[int(v2)][int(v1)] = int(weight)

	return result
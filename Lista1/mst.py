import parser as par

import sys

MAX_WEIGHT = 9999999999
result = []

def minimum_key(key: [int], mst_set: [bool], size: int):
	min = MAX_WEIGHT
	min_index = 0
	for v in range(0, size):
		if mst_set[v] == False and key[v] < min:
			min = key[v]
			min_index = v;

	return min_index

def MST(parent: [int], graph: [[int]], size: int):
	v = [[] * size] * size
	for i in range(1, size):
		p = [] * size
		p.append(parent[i])
		p.append(i)
		v.append(p)
		p.clear()

	return v

# def DFS():

def printMST(parent, graph, size):
    print("Edge \tWeight")
    for i in range(1, size):
        print(parent[i], "-", i, "\t", graph[i][parent[i]])


def prim_MST(graph: [[int]], size: int):
	parent = [0] * size
	key = [MAX_WEIGHT] * size
	mst_set = [False] * size

	key[0] = 0
	parent[0] = -1

	for count in range(0, size - 1):
		u = minimum_key(key, mst_set, size)
		mst_set[u] = True
		for v in range(0, size):
			if graph[u][v] > 0 and mst_set[v] == False and graph[u][v] < key[v]:
				parent[v] = u
				key[v] = graph[u][v]

	v = MST(parent, graph, size)
	return v

def main() -> int:
	result = par.parse(str(sys.argv[1]))
	v = prim_MST(result, len(result))
	# print(v)

if __name__ == '__main__':
    main()
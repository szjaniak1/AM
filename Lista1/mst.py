import parser as par

import sys

MAX_WEIGHT = 9999999999
final_ans = []

def minimum_key(key: [int], mst_set: [bool], size: int):
	min = MAX_WEIGHT
	min_index = 0
	for v in range(0, size):
		if mst_set[v] == False and key[v] < min:
			min = key[v]
			min_index = v;

	return min_index

def MST(parent: [int], graph: [[int]], size: int):
	v = [[]]
	for i in range(1, size):
		p = []
		p.append(parent[i])
		p.append(i)
		v.append(p)

	return v

def DFS(edges_list: [[int]], num_nodes: int, starting_vertex: bool, visited_nodes: [bool]):
	final_ans.append(starting_vertex)
	visited_nodes[starting_vertex] = True

	for i in range(num_nodes):
		if i == starting_vertex:
			continue
		if edges_list[starting_vertex][i] == 1:
			if visited_nodes[i]:
				continue
			DFS(edges_list, num_nodes, i, visited_nodes)


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
			if mst_set[v] == False and graph[u][v] < key[v]:
				parent[v] = u
				print(u)
				key[v] = graph[u][v]

	print(parent)

	v = MST(parent, graph, size)
	return v

def main() -> int:
	result = par.parse(str(sys.argv[1]))
	size = len(result)
	v = prim_MST(result, size)

	visited_nodes = [False] * size
	edges_list = [[0 for i in range(size)] for j in range(size)]
	for i in range(1, len(v)):
		first_node = v[i][0]
		second_node = v[i][1]
		edges_list[first_node][second_node] = 1
		edges_list[second_node][first_node] = 1

	DFS(edges_list, size, 0, visited_nodes)
	final_ans.append(final_ans[0])

	# for i in range(0, len(final_ans)):
	# 	print(str(final_ans[i]) + "-")

if __name__ == '__main__':
    main()
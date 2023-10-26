import parser as par

import sys
import tsplib95 as tsp
import networkx as nx

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

if __name__ == '__main__':
    main()
import parser as par

import sys
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from numpy.random import shuffle
import numpy.random as random

MAX_WEIGHT = 9999999999
tsp_permutation = []

def get_path_cost(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]

    distance += graph[path[0]][path[-1]]

    return distance

def minimum_key(key: [int], mst_set: [bool], size: int):
	min = MAX_WEIGHT
	min_index = 0
	for v in range(0, size):
		if mst_set[v] == False and key[v] < min:
			min = key[v]
			min_index = v;

	return min_index

def MST(parent: [int], size: int):
	v = []
	for i in range(1, size):
		p = []
		p.append(parent[i])
		p.append(i)
		v.append(p)

	return v

def DFS(edges_list: [[int]], num_nodes: int, starting_vertex: int, visited_nodes: [bool]):
	tsp_permutation.append(starting_vertex)
	visited_nodes[starting_vertex] = True

	for i in range(num_nodes):
		if i == starting_vertex:
			continue
		if edges_list[starting_vertex][i] == 1:
			if visited_nodes[i]:
				continue
			DFS(edges_list, num_nodes, i, visited_nodes)

def weight_TSP(tsp, graph, size):
    weight = 0
    for i in range(0, size - 1):
    	weight += graph[tsp[i]][tsp[i + 1]]

    return weight

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

	mst = MST(parent, size)
	return mst, parent

def plot_MST(graph: [[int]], points: [par.Point], file_name: str, weight: int):
	for i in range(0, len(graph)):
		v1, v2 = graph[i]
		p1 = [points[v1].pos_x, points[v1].pos_y]
		p2 = [points[v2].pos_x, points[v2].pos_y]
		x_values = [p1[0], p2[0]]
		y_values = [p1[1], p2[1]]
		plt.plot(x_values, y_values, marker='.', color='b')

	plt.title(f'{file_name} - weight: {weight}')
	plt.savefig(f'./graphs/MST_{file_name}')
	plt.close()

def plot_TSP(tsp: [int], points: [par.Point], file_name: str, weight: int):
	data = []
	for i in tsp:
		p = [points[i].pos_x, points[i].pos_y]
		data.append(p)

	data = np.array(data)
	plt.plot(data[:, 0], data[:, 1], marker='.', color='r')
	plt.title(f'{file_name} - weight: {weight}')
	plt.savefig(f'./graphs/TSP_{file_name}')
	plt.close()

def invert_weight(permutation, adj_matrix, i, j, weight):
    last = len(permutation) - 1
    pre = i - 1 if i > 0 else last
    post = (j + 1) % len(permutation)

    return (
        weight
        - adj_matrix[permutation[i]][permutation[pre]]
        - adj_matrix[permutation[j]][permutation[post]]
        + adj_matrix[permutation[j]][permutation[pre]]
        + adj_matrix[permutation[i]][permutation[post]]
    )

def get_neighbourhood(permutation, adj_matrix, weight):
    neighborhood = []
    length = len(permutation)

    for diff in range(1, length // 2):
        for j in range(diff, length):
            neighborhood.append((j - diff, j, invert_weight(permutation, adj_matrix, j - diff, j, weight)))

    return neighborhood

def local_search(permutation: [int], graph: [[int]], edges_list: [[int]]):
	curr_weight = weight_TSP(permutation, graph, len(permutation))
	curr = permutation
	counter = 0

	while True:
		counter += 1
		neighbourhood = get_neighbourhood(curr, graph, curr_weight)
		candidate = min(neighbourhood, key=lambda x: x[2])
		if candidate[2] >= curr_weight:
			break
		start, end, _ = candidate
		curr[start:end+1] = reversed(curr[start:end + 1])
		curr_weight = candidate[2]

	return curr, counter, curr_weight

def convert_MST_to_adjacency_matrix(mst: [[int]]):
	size = len(mst)
	edges_list = [[0 for i in range(size + 1)] for j in range(size + 1)]
	for i in range(size):
		first_node = mst[i][0]
		second_node = mst[i][1]
		edges_list[first_node][second_node] = 1
		edges_list[second_node][first_node] = 1

	return edges_list

def main():
	for file_name in os.listdir('data'):
		file_name = file_name[:-4]
		graph, points = par.parse(f'./data/{file_name}.tsp')

		mst, parent = prim_MST(graph, len(graph))

		edges_list = convert_MST_to_adjacency_matrix(mst)
		n = math.sqrt(len(points) - 1)

		dfs_steps = 0
		dfs_mean = 0
		dfs_min = MAX_WEIGHT
		min_permutation = []
		print(file_name)
		for i in range(100):
			tsp_permutation.clear()
			rand_point = random.randint(1, len(points) - 1)
			visited_nodes = [False] * len(mst)
			DFS(edges_list, len(mst), rand_point, visited_nodes)
			tsp_permutation.append(rand_point)
			p, counter, w = local_search(tsp_permutation, graph, edges_list)
			print(w)
			dfs_mean += w
			dfs_steps += counter
			if w < dfs_min:
				dfs_min = w
				min_permutation = p
				print(min_permutation)

		plot_TSP(min_permutation, points, file_name + "_loc1", int(dfs_min))
		print("min")
		print(dfs_min)

		# weight_tsp = weight_TSP(tsp_permutation, result, len(tsp_permutation))
		# plot_TSP(tsp_permutation, points, file_name, int(weight_tsp))

if __name__ == '__main__':
    main()
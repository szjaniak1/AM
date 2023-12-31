import parser as par

import sys
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from numpy.random import shuffle

MAX_WEIGHT = 9999999999
final_ans = []

def get_path_cost(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]

    distance += graph[path[0]][path[-1]]

    return distance

def get_minimums(graph, path):
    path = path[:-1]

    min_10_list_y = []
    min_10_list_x = []
    min_50_list_y = []
    min_50_list_x = []
    min_10_x = 0
    min_10_y = math.inf
    min_50_x = 0
    min_50_y = math.inf
    min_1000_x = 0
    min_1000_y = math.inf
    for i in range(1, 1001):
        shuffle(path)

        cost = get_path_cost(graph, path)
        if cost < min_10_y:
            min_10_x = i
            min_10_y = cost
        if cost < min_50_y:
            min_50_x = i
            min_50_y = cost
        if cost < min_1000_y:
            min_1000_x = i
            min_1000_y = cost
        if i % 10 == 0:
            min_10_list_y.append(min_10_y)
            min_10_list_x.append(min_10_x)
            min_10_y = math.inf
        if i % 50 == 0:
            min_50_list_y.append(min_50_y)
            min_50_list_x.append(min_50_x)
            min_50_y = math.inf
    min_10_list = [min_10_list_x, min_10_list_y]
    min_50_list = [min_50_list_x, min_50_list_y]
    min_1000 = [min_1000_x, min_1000_y]
    return min_10_list, min_50_list, min_1000


def plot_data(min_10_list, min_50_list, min_1000, file_name):

    plt.scatter(min_1000[0], min_1000[1], c="red", s=150, label='1000 Minimum')

    plt.scatter(min_50_list[0], min_50_list[1],c="orange", s= 40, label='50 Minimum')

    plt.scatter(min_10_list[0], min_10_list[1], c="blue", s = 8,label='10 Minimum')

    plt.title(file_name)

    plt.legend()

    plt.grid(True)
    plt.savefig(f"./graphs/min_{file_name}")
    plt.close()

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
	final_ans.append(starting_vertex)
	visited_nodes[starting_vertex] = True

	for i in range(num_nodes):
		if i == starting_vertex:
			continue
		if edges_list[starting_vertex][i] == 1:
			if visited_nodes[i]:
				continue
			DFS(edges_list, num_nodes, i, visited_nodes)

def weight_MST(parent, graph, size):
    weight = 0
    for i in range(1, size):
    	weight += graph[i][parent[i]]

    return weight

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
		result, points = par.parse(f'./data/{file_name}.tsp')
		mst, parent = prim_MST(result, len(result))
		weight = weight_MST(parent, result, len(result))
		plot_MST(mst, points, file_name, int(weight))

		edges_list = convert_MST_to_adjacency_matrix(mst)
		visited_nodes = [False] * len(mst)
		DFS(edges_list, len(mst), 0, visited_nodes)

		weight_tsp = weight_TSP(final_ans, result, len(final_ans))
		plot_TSP(final_ans, points, file_name, int(weight_tsp))

		min_10_list, min_50_list, min_1000 = get_minimums(result, final_ans)
		plot_data(min_10_list, min_50_list, min_1000, file_name)
		final_ans.clear()

if __name__ == '__main__':
    main()
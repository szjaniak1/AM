import parser as par

import sys
import numpy as np
import os
from matplotlib import pyplot as plt

MAX_WEIGHT = 9999999999

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

# def DFS():

def weight_MST(parent, graph, size):
    weight = 0
    for i in range(1, size):
    	weight += graph[i][parent[i]]

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

	adj_list = MST(parent, size)
	return adj_list, parent

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

def main() -> int:
	for file_name in os.listdir('data'):
		file_name = file_name[:-4]
		result, points = par.parse(f'./data/{file_name}.tsp')
		mst, parent = prim_MST(result, len(result))
		weight = weight_MST(parent, result, len(result))
		plot_MST(mst, points, file_name, int(weight))

if __name__ == '__main__':
    main()
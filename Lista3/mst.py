import parser as par

import sys
import numpy as np
import os
import math
import numpy.random as rand
import random
import copy

from multiprocessing import Process
from matplotlib import pyplot as plt
from numpy.random import shuffle

MAX_WEIGHT = 9999999999

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

def weight_MST(parent, graph, size):
    weight = 0
    for i in range(1, size):
        weight += graph[i][parent[i]]

    return weight

def DFS(tsp_permutation: [int], edges_list: [[int]], num_nodes: int, starting_vertex: int, visited_nodes: [bool]):
    tsp_permutation.append(starting_vertex)
    visited_nodes[starting_vertex] = True

    for i in range(num_nodes):
        if i == starting_vertex:
            continue
        if edges_list[starting_vertex][i] == 1:
            if visited_nodes[i]:
                continue
            DFS(tsp_permutation, edges_list, num_nodes, i, visited_nodes)

def convert_MST_to_adjacency_matrix(mst: [[int]]):
    size = len(mst)
    edges_list = [[0 for i in range(size + 1)] for j in range(size + 1)]
    for i in range(size):
        first_node = mst[i][0]
        second_node = mst[i][1]
        edges_list[first_node][second_node] = 1
        edges_list[second_node][first_node] = 1

    return edges_list

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

def permutation_weight(permutation: [int], adj_matrix: [[int]]):
    s = 0
    prev = permutation[0]
    for cur in permutation[1:]:
        s += adj_matrix[prev][cur]
        prev = cur

    s += adj_matrix[0][-1]
    return s

def get_random_permutation(point_count: int):
    permutation = list(range(point_count))
    rand.shuffle(permutation)
    return permutation

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

def get_neighbourhood(permutation, adj_matrix):
    weight = permutation_weight(permutation, adj_matrix)
    neighborhood = []
    length = len(permutation)

    for diff in range(1, length // 2):
        for j in range(diff, length):
            neighborhood.append((j - diff, j, invert_weight(permutation, adj_matrix, j - diff, j, weight)))

    return neighborhood

def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbor = solution[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def points_to_matrix(points: [[int]]):
    point_count = len(points)
    adj_matrix = [[0] * point_count for _ in range(point_count)]

    for i in range(point_count):
        for j in range(i, point_count):
            if j != i:
                p1 = points[i]
                p2 = points[j]
                dist = round(((p1.pos_x - p2.pos_x) ** 2 + (p1.pos_y - p2.pos_y) ** 2) ** 0.5)
                adj_matrix[i][j] = dist
                adj_matrix[j][i] = dist

    return adj_matrix

def simulated_annealing(adj_matrix, alpha, beta, gamma, delta):
    point_count = len(adj_matrix)
    temperature = int(point_count * alpha)
    _epoch_range = int(temperature * delta)
    max_iterations = int(point_count * gamma)
    solution = get_random_permutation(point_count)
    current_weight = permutation_weight(solution, adj_matrix)

    for i in range(max_iterations):
        for _epoch in range(_epoch_range):
            swap_idx = random.sample(range(point_count), 2)
            potential_solution = solution[:]
            potential_solution[swap_idx[0]], potential_solution[swap_idx[1]] = (
                potential_solution[swap_idx[1]],
                potential_solution[swap_idx[0]],
            )
            potential_weight = permutation_weight(potential_solution, adj_matrix)

            if potential_weight < current_weight:
                current_weight = potential_weight
                solution = potential_solution
            elif random.random() < math.exp((current_weight - potential_weight) / temperature):
                current_weight = potential_weight
                solution = potential_solution

        temperature *= beta

    return solution, current_weight

def tabu_search(initial_solution, adj_matrix, alpha, beta):
    point_count = len(adj_matrix)
    tabu_list_size = int(point_count * alpha)
    max_iterations = int(point_count * beta)

    best_solution = initial_solution
    current_solution = initial_solution
    tabu_list = []
 
    for _ in range(max_iterations):
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_fitness = 9999999
 
        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_fitness = permutation_weight(neighbor, adj_matrix)
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
 
        if best_neighbor is None:
            break
 
        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
 
        if permutation_weight(best_neighbor, adj_matrix) < permutation_weight(best_solution, adj_matrix):
            best_solution = best_neighbor
 
    return best_solution, permutation_weight(best_solution, adj_matrix)

def main():
    for file_name in os.listdir('data'):
        file_name = file_name[:-4]
        print(file_name)

        graph, points, points_count = par.parse(f'./data/{file_name}.tsp')
        adj_matrix = points_to_matrix(points)

        mst, parent = prim_MST(graph, len(graph))
        weight_mst = weight_MST(parent, graph, len(graph))
        edges_list = convert_MST_to_adjacency_matrix(mst)
        n = len(points) - 1
        tsp_permutation = []
        visited_nodes = [False] * len(mst)
        DFS(tsp_permutation, edges_list, len(mst), 0, visited_nodes)
        tsp_permutation.append(tsp_permutation[0])

        sa, sa_best_weight = simulated_annealing(adj_matrix, alpha=0.5, beta=0.95, delta=0.3, gamma=0.3)
        print(sa_best_weight)

        tabu, tabu_best_weight = tabu_search(tsp_permutation, adj_matrix, alpha=0.1, beta=0.2)
        print(tabu_best_weight)

        result_file = open("./results/" + file_name + "_result", "a")
        result_file.write("\nsimulated_annealing : " + str(sa_best_weight) + "\ntabu_search : " + str(tabu_best_weight) + "\n")
        result_file.close()

if __name__ == '__main__':
    main()
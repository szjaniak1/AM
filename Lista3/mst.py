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

def simulated_annealing(adj_matrix, temperature):
    point_count = len(adj_matrix)
    solution = get_random_permutation(point_count)
    current_weight = permutation_weight(solution, adj_matrix)
    print("SEED SOLUTION:", current_weight)

    while temperature != 0:
        for _epoch in range(10_000):
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

        temperature -= 1

    return solution, current_weight

def tabu_search(adj_matrix):
    tabu_list_size = len(adj_matrix)
    max_iterations = 200

    initial_solution = get_random_permutation(tabu_list_size)
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
    file_name = "xqf131"
    raph, points, points_count = par.parse(f'./data/{file_name}.tsp')
    adj_matrix = points_to_matrix(points)
    sa, best_weight = simulated_annealing(adj_matrix, points_count)
    print(best_weight)

    tabu, best_weight = tabu_search(adj_matrix)
    print(best_weight)

    # for file_name in os.listdir('data'):
    #     file_name = file_name[:-4]

    #     graph, points, points_count = par.parse(f'./data/{file_name}.tsp')
    #     adj_matrix = points_to_matrix(points)
    #     # sa, best_weight = simulated_annealing(adj_matrix, points_count)
    #     # print(sa)
    #     # print(best_weight)

    #     tabu, best_weight = tabu_search(adj_matrix)
    #     print(tabu)
    #     print(best_weight)

if __name__ == '__main__':
    main()
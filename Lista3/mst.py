import parser as par

import sys
import numpy as np
import os
import math
import numpy.random as rand
import random

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

def get_neighbourhood(permutation, adj_matrix, weight):
    neighborhood = []
    length = len(permutation)

    for diff in range(1, length // 2):
        for j in range(diff, length):
            neighborhood.append((j - diff, j, invert_weight(permutation, adj_matrix, j - diff, j, weight)))

    return neighborhood

def local_search(permutation: [int], graph: [[int]]):
	curr_weight = weight_TSP(permutation, graph, len(permutation))
	curr = permutation.copy()
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
            elif random.random() < math.exp(
                (current_weight - potential_weight) / temperature
            ):
                current_weight = potential_weight
                solution = potential_solution

        temperature -= 1

    return solution, current_weight

def main():
	for file_name in os.listdir('data'):
		file_name = file_name[:-4]

		graph, points, points_count = par.parse(f'./data/{file_name}.tsp')
		adj_matrix = points_to_matrix(points)
		sa = simulated_annealing(adj_matrix, points_count)
		print(sa)

if __name__ == '__main__':
    main()
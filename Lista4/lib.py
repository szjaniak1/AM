import random
import numpy as np

from copy import deepcopy

Point = tuple[float, float]

class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

class Evolution:
    def __init__(self, island_count, point_count, adj_matrix):
        self.islands = []
        self.adj_matrix = adj_matrix
        self.rng = np.random.default_rng()
        mst = gen_mst(adj_matrix, point_count)
        fitness_mst = self.permutation_weight(mst, adj_matrix)
        mst_individual = Individual(list(mst), fitness_mst)
        chromosome = list(range(point_count))
        for _ in range(island_count):
            island = []
            island.append(deepcopy(mst_individual))
            for _ in range(64):
                random.shuffle(chromosome)
                fitness = self.permutation_weight(chromosome, adj_matrix)
                island.append(Individual(deepcopy(chromosome), fitness))
            self.islands.append(island)

    def run(self, pmx):
        generation = 0
        no_improvement = 0
        curr_best = self.extract_best()
        while True:
            self.selection()
            if generation % 100 == 0:
                self.migration()
            self.reproduction(pmx)
            generation += 1
            new_best = self.extract_best()
            if new_best[1] < curr_best[1]:
                curr_best = new_best
                no_improvement = 0
            else:
                no_improvement += 1
            if generation == 100 or no_improvement == 10:
                break

    def extract_best(self):
        best = deepcopy(self.islands[0][0])
        for island in self.islands:
            for individual in island:
                if individual.fitness < best.fitness:
                    best = deepcopy(individual)
        return list(best.chromosome), best.fitness

    def reproduction(self, pmx):
        for op in self.islands:
            new_population = []
            for _ in range(10):
                parents = random.sample(op, 2)
                child1, child2 = self.cx_crossover(parents[0].chromosome, parents[1].chromosome)
                fitness1 = self.permutation_weight(child1, self.adj_matrix)
                fitness2 = self.permutation_weight(child2, self.adj_matrix)
                
                if random.random() < 0.1:
                    self.mutation(child1)
                if random.random() < 0.1:
                    self.mutation(child2)

                if len(new_population) == 100:
                    break

                new_population.append(Individual(chromosome=child1, fitness=fitness1))
                new_population.append(Individual(chromosome=child2, fitness=fitness2))

            op[:] = new_population


    def selection(self):
        for island in self.islands:
            new_population = []
            for _ in range(10):
                tournament = random.sample(island, 5)
                tournament.sort(key=lambda x: x.fitness)
                new_population.append(deepcopy(tournament[0]))
            island[:] = new_population

    def migration(self):
        island_count = len(self.islands)
        for _ in range(island_count):
            source_island = random.randint(0, island_count - 1)
            target_island = random.randint(0, island_count - 1)
            if source_island != target_island and self.islands[source_island]:
                individual_index = random.randint(0, len(self.islands[source_island]) - 1)
                individual = self.islands[source_island].pop(individual_index)
                self.islands[target_island].append(individual)

    @staticmethod
    def cx_crossover(parent1, parent2):
        length = len(parent1)
        child1 = [None] * length
        child2 = [None] * length

        index = 0
        cycle = 0
        while cycle < length:
            while child1[index] is None:
                child1[index] = parent1[index]
                child2[index] = parent2[index]
                index = parent2.index(parent1[index])
            cycle += 1
            index = (index + 1) % length

        for i in range(length):
            if child1[i] is None:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        return child1, child2

    @staticmethod
    def mutation(child):
        if len(child) < 2:
            return
        start = random.randint(0, len(child) - 2)
        end = random.randint(start + 1, len(child) - 1)
        child[start:end] = reversed(child[start:end])

    @staticmethod
    def permutation_weight(permutation, adj_matrix):
        s = 0
        prev = permutation[0]
        for cur in permutation[1:]:
            s += adj_matrix[prev][cur]
            prev = cur

        s += adj_matrix[0][-1]
        return s

def read_lines(filename):
    with open(filename) as file:
        return file.readlines()

def file_to_points(filename):
    points = []
    lines = read_lines(filename)[8:]
    for line in lines:
        if line == "EOF\n":
            break
        tmp = line.split()
        points.append((float(tmp[1]), float(tmp[2])))
    return points

def points_to_matrix(points):
    point_count = len(points)
    adj_matrix = np.zeros((point_count, point_count), dtype=int)
    for i in range(point_count):
        for j in range(i, point_count):
            if j != i:
                p1, p2 = points[i], points[j]
                dist = round(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
                adj_matrix[i][j] = dist
                adj_matrix[j][i] = dist
    return adj_matrix.tolist()

def gen_mst(adj_matrix, point_count):
    parent = prim(adj_matrix, point_count)
    mst = parent_to_adj_list(parent)
    return dfs(mst)

def parent_to_adj_list(parent):
    adj_list = [[] for _ in parent]
    for u, v in enumerate(parent[1:], start=1):
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list

def dfs(graph):
    visited = []
    traversal = []
    for i in range(len(graph)):
        if i not in visited:
            stack = []
            visited.append(i)
            stack.append(i)
            while stack:
                node = stack.pop()
                traversal.append(node)
                for j in graph[node]:
                    if j not in visited:
                        visited.append(j)
                        stack.append(j)
    return traversal

def prim(adj_matrix, point_count):
    parent = [float('inf')] * point_count
    key = [float('inf')] * point_count
    mst_set = [False] * point_count

    key[0] = 0

    for _ in range(point_count - 1):
        u = min_key(key, mst_set)
        mst_set[u] = True
        for v in range(point_count):
            if adj_matrix[u][v] != 0 and not mst_set[v] and adj_matrix[u][v] < key[v]:
                parent[v] = u
                key[v] = adj_matrix[u][v]

    return parent

def min_key(key, mst_set):
    minimum = float('inf')
    min_index = 0
    for v in range(len(key)):
        if not mst_set[v] and key[v] < minimum:
            minimum = key[v]
            min_index = v
    return min_index
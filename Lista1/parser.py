from dataclasses import dataclass
from math import sqrt

@dataclass
class Point:
	pos_x: int
	pos_y: int

def parse(path: str):
	file = open(path)
	dim, dim_num = file.readline().split(':')
	size = int(dim_num) + 1
	points = [Point(0, 0) for i in range(size)]

	for line in file:
		num, x, y = line.split(' ')
		points[int(num)] = Point(int(x), int(y))

	return calculate_weight(points)

def calculate_weight(points: [Point]):
	size = len(points)
	result = [[0 for i in range(len(points))] for j in range(len(points))]

	for i in range(size):
		for j in range(size):
			result[i][j] = get_weight(points[i], points[j])

	return result

def get_weight(point1: Point, point2: Point):
	return sqrt(pow((point1.pos_x - point2.pos_x), 2) + pow((point1.pos_y - point2.pos_y), 2))
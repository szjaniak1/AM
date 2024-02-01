import threading
import time
import numpy as np

from lib import Evolution, file_to_points, points_to_matrix

def main():
    points = file_to_points("test_data/1.tsp")
    point_count = len(points)
    adj_matrix = points_to_matrix(points)
    avg_time = [0.0]
    avg_weight = [0.0]

    def evolution_thread(pmx):
        nonlocal avg_time, avg_weight
        for _ in range(10):
            start = time.time()
            ga = Evolution(4, point_count, adj_matrix.copy())
            ga.run(pmx)
            _, weight = ga.extract_best()
            elapsed = time.time() - start
            with lock:
                avg_time[0] += elapsed / 10.0
                avg_weight[0] += weight / 10.0

    lock = threading.Lock()

    with open("data.csv", "w") as file:
        file.write("map;avg_weight;avg_time\n")
        for path in [
            "test_data/1.tsp",
            "test_data/2.tsp",
            "test_data/3.tsp",
            "test_data/4.tsp",
            "test_data/5.tsp",
            "test_data/6.tsp",
            "test_data/7.tsp",
            "test_data/8.tsp",
            "test_data/9.tsp",
            "test_data/a.tsp",
            "test_data/b.tsp",
            "test_data/c.tsp",
            "test_data/d.tsp",
            "test_data/e.tsp",
            "test_data/f.tsp",
        ]:
            points = file_to_points(path)
            point_count = len(points)
            adj_matrix = points_to_matrix(points)

            avg_time = [0.0]
            avg_weight = [0.0]
            for _ in range(10):
                threads = []
                for _ in range(10):
                    thread = threading.Thread(target=evolution_thread, args=(True,))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

            file.write(f"{point_count};{avg_weight[0]};{avg_time[0]}\n")

if __name__ == "__main__":
    main()

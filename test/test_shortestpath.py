import numpy as np
from src.shortestpath import Agent


def test_params():
    rows= 5
    columns = 5
    seed = 5
    np.random.seed(seed)

    results = np.array([0])
    grid = np.random.randint(0, 9, size=(rows, columns))


    my = Agent(grid, seed)

    iterations, shortest_path, path_length = my.RandomSearch()
    results = np.append(results,np.array([[rows,columns,"Random",iterations,path_length]]))

    iterations, shortest_path, path_length  = my.SimpleSearch()
    results = np.append(results, np.array([[rows, columns, "Random", iterations, path_length]]))

    iterations, shortest_path, path_length  = my.FastSearch()
    results = np.append(results, np.array([[rows, columns, "Random", iterations, path_length]]))

    iterations, shortest_path =my.DijkstrasSearch()
    results = np.append(results, np.array([[rows, columns, "Random", iterations, path_length]]))
    print(results)

test_params()



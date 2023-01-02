import numpy as np
import matplotlib.pyplot as plt
from src.shortestpath import Agent

'''
This function can be called to test the different search 
algorithms with a specific set of rows and columns
'''

def test_game(rows,columns,seed):

    np.random.seed(seed)

    grid = np.random.randint(0, 9, size=(rows, columns))

    cell_count = rows*columns
    print("Game Floor", grid)

    my = Agent(grid, seed)

    master_metrics_log = np.empty((0,4),float)

    iterations, shortest_path, path_length = my.RandomSearch()
    master_metrics_log = np.append(master_metrics_log, np.array([["RandomSearch", cell_count, iterations, path_length]]),0)
    print("For Random Search: \n", "Iterations: ", iterations, "\n Shortest Path: ", shortest_path, "with length",
          path_length)

    iterations, shortest_path, path_length = my.SimpleSearch()
    master_metrics_log = np.append(master_metrics_log, np.array([["SimpleSearch", cell_count, iterations, path_length]]),0)
    print("For Simple Search: \n", "Iterations: ", iterations, "\n Shortest Path: ", shortest_path, "with length",
          path_length)

    iterations, shortest_path, path_length = my.FastSearch()
    master_metrics_log = np.append(master_metrics_log, np.array([["FastSearch", cell_count, iterations, path_length]]),0)
    print("For Fast Search: \n", "Iterations: ", iterations, "\n Shortest Path: ", shortest_path, "with length",
          path_length)

    iterations, shortest_path, path_length = my.DijkstrasSearch()
    master_metrics_log = np.append(master_metrics_log, np.array([["Dijkstras", cell_count, iterations, path_length]]),0)
    print("For Dijkstras: \n", "Iterations: ", iterations, "\n Shortest Path: ", shortest_path, "with length",
          path_length)
    return master_metrics_log


'''
This test function iterates over a set of grid sizes and tests the 
different search algorithms
'''
def test_grid_size():

    master_metrics_log = np.empty((0,4),float)
    for rows in range (3,10):
        for cols in range(3,10):
            metrics = test_game(rows,cols,5)
            print(metrics)
            master_metrics_log = np.append(master_metrics_log,metrics,0)

    print(master_metrics_log)

    print("Random Search Average Iterations:",master_metrics_log[master_metrics_log[:, 0] == "RandomSearch"][:, 2].astype(float).mean(axis=0))
    print("Simple Search Average Iterations:",master_metrics_log[master_metrics_log[:, 0] == "SimpleSearch"][:, 2].astype(float).mean(axis=0))
    print("Fast Search Average Iterations:",master_metrics_log[master_metrics_log[:, 0] == "FastSearch"][:, 2].astype(float).mean(axis=0))
    print("Dijkstras Search Average Iterations:",master_metrics_log[master_metrics_log[:, 0] == "Dijkstras"][:, 2].astype(float).mean(axis=0))

    print("Random Search Average Path Length:",master_metrics_log[master_metrics_log[:, 0] == "RandomSearch"][:, 3].astype(float).mean(axis=0))
    print("Simple Search Average Path Length:",master_metrics_log[master_metrics_log[:, 0] == "SimpleSearch"][:, 3].astype(float).mean(axis=0))
    print("Fast Search Average Path Length:",master_metrics_log[master_metrics_log[:, 0] == "FastSearch"][:, 3].astype(float).mean(axis=0))
    print("Dijkstras Search Average Path Length:",master_metrics_log[master_metrics_log[:, 0] == "Dijkstras"][:, 3].astype(float).mean(axis=0))


    plt.figure(figsize=(16,12))
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="RandomSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="RandomSearch"][:, 2].astype(float),label="RandomSearch")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="SimpleSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SimpleSearch"][:, 2].astype(float),label="SimpleSearch")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="FastSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="FastSearch"][:, 2].astype(float),label="FastSearch")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="Dijkstras"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Dijkstras"][:, 2].astype(float),label="Dijkstras")
    plt.xlabel("Cell Count")
    plt.ylabel("Number of Iterations")
    plt.legend()
    plt.show()

    figure, axs = plt.subplots(2,2)
    plt.figure(figsize=(20,15))
    axs[0,0].plot(master_metrics_log[master_metrics_log[:,0]=="RandomSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="RandomSearch"][:, 3].astype(float),label="RandomSearch")
    axs[0,0].set_title("RandomSearch")
    axs[0,1].plot(master_metrics_log[master_metrics_log[:,0]=="SimpleSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SimpleSearch"][:, 3].astype(float),label="SimpleSearch")
    axs[0,1].set_title("SimpleSearch")
    axs[1,0].plot(master_metrics_log[master_metrics_log[:,0]=="FastSearch"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="FastSearch"][:, 3].astype(float),label="FastSearch")
    axs[1,0].set_title("FastSearch")
    axs[1,1].plot(master_metrics_log[master_metrics_log[:,0]=="Dijkstras"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Dijkstras"][:, 3].astype(float),label="Dijkstras")
    axs[1,1].set_title("Dijkstras")

    plt.show()

test_grid_size()







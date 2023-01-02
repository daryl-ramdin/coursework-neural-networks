import numpy as np


class Node:
    def __init__(self, row,col):
        self.row = row
        self.id = str(row)+","+str(col)
        self.col = col
        self.nearest_neighbour = None

    def set_neighbour(self, neighbour):
        self.nearest_neighbour = neighbour

def find_node(node_list, row, col):
    id = str(row)+","+str(col)
    for node in node_list:
        if node.id == id: return node
    return None

class Agent:
    '''
    This represents that agent that has to traverse the grid in the shortest possible time.
    The grid is represented by the member variable called game_floor which is a r by c grid
    '''

    def __init__(self, game_floor, seed):
        '''

        :param game_floor: this is a copy of the grid that the agent has to traverse
        '''
        self.game_floor = game_floor
        self.max_row = game_floor.shape[0] - 1
        self.max_col = game_floor.shape[1] - 1
        self.last_grid = [game_floor.shape[0] - 1, game_floor.shape[1] - 1]
        self.shortest_path = []
        self.iterations = 0
        self.seed = seed

    def FastSearch(self):
        '''
        This search is an improvement to the Simple Search. It is based on Dijkstra's algorithm
        that says to get from A to Z we will have to calculate a subpath which is also the shortest path between
        its source and destination (https://www.analyticssteps.com/blogs/dijkstras-algorithm-shortest-path-algorithm).
        Based on this, the Simple Search is refined so that starting from the source node, we find the shortest path
        from each of it's left and bottom adjacent nodes. We then take that node and find the shortest from its adjacent
        nodes. This is done repeatedly until we reach the destination node.
        '''

        # Initialise the first cell and iteration count
        row = col = 0
        i = 0
        current_cell = [0, 0]
        self.iterations = 0

        pathlength, shortest_path = self.get_pathlength(current_cell)

        # Remove the topmost row as that contains the cell at [0,0]
        print("Shortest path length will take ", pathlength, " seconds")
        shortest_path = np.reshape(shortest_path, [int(len(shortest_path) / 2), 2])
        print("Following this path: ")
        shortest_path = np.delete(shortest_path,0,0)
        return self.iterations, shortest_path, self.get_total_pathlength(shortest_path)

    def get_pathlength(self, current_cell):
        '''
        :param self:
        :param i: the row of the current cell
        :param j: the column of the current cell
        :return:
        '''
        self.iterations += 1

        if current_cell == None:
            return float("inf"), None

        my_path = np.array(current_cell)

        i = current_cell[0]
        j = current_cell[1]

        if current_cell == [0, 0]:
            my_path_length = 0
        else:
            my_path_length = self.game_floor[i, j]

        bottom_cell = right_cell = None

        if i == self.max_row:
            # We're at the bottom row and can't go any further down. We set the bottom cell to None
            bottom_cell = None
        elif i < self.max_row:
            # Let's move one cell down
            bottom_cell = [i + 1, j]

        if j == self.max_col:
            # We're at the rightmost cell and can't go any further right. We set right cell to None
            right_cell = None
        else:
            # Let's move to the right
            right_cell = [i, j + 1]

        bottom_path = right_path = []
        bottom_length = right_length = float("inf")
        if bottom_cell == self.last_grid or right_cell == self.last_grid:
            # The next cell is the end cell so return our length
            # print("The end cell")
            # print(my_path)
            my_path = np.insert(my_path, axis=0, values=self.last_grid, obj=2)
            # print(my_path)
        else:
            # Let's continue searching
            if bottom_cell != None:
                # print ("Move Bottom")
                bottom_length, bottom_path = self.get_pathlength(bottom_cell)
            if right_cell != None:
                # print("Move Right")
                right_length, right_path = self.get_pathlength(right_cell)

            if bottom_length <= right_length:
                # print("Move Bottom")
                # print(bottom_path)
                my_path_length = my_path_length + bottom_length
                my_path = np.insert(my_path, axis=0, values=bottom_path, obj=len(my_path))
            else:
                # print("Move Right")
                # print (right_path)
                my_path_length = my_path_length + right_length
                my_path = np.insert(my_path, axis=0, values=right_path, obj=len(my_path))
        # print("MyPath",my_path)
        return my_path_length, my_path

    def SimpleSearch(self):
        '''
        For the simple search we keep comparing the bottom and right adjacent cells
        and moving to the cell with the shortest path. If both the right and bottom
        have the same length, we move right.
        :return:
        '''
        path_array = []
        self.iterations = 0
        # Set the last along with the maximum row and column count
        last_grid = [self.game_floor.shape[0] - 1, self.game_floor.shape[1] - 1]
        max_row = self.game_floor.shape[0] - 1
        max_col = self.game_floor.shape[1] - 1

        # Initialise the first cell and iteration count
        row = col = 0
        i = 0
        current_cell = [0, 0]

        # Let's start our search until we reach the end
        while last_grid != current_cell:
            self.iterations += 1
            if (row + 1) > max_row:
                # We're at the bottom of the grid so we move right
                col += 1
                # print ("Move right")
            elif (col + 1) > max_col:
                # We're at the right of the grid so move down
                row += 1
                # print("Move down")
            elif self.game_floor[row + 1, col] < self.game_floor[row, col + 1]:
                # The lower cell is faster so move down
                row += 1
                # print("Move down")
            elif self.game_floor[row, col + 1] <= self.game_floor[row + 1, col]:
                # The cell to the right is faster or the same so move right
                col += 1
                # print("Move right")
            else:
                # This does nothing
                i += 1

            current_cell = [row, col]
            path_array.append(current_cell)
            # print ("Current cell is {0}".format(current_cell))
        return self.iterations, np.array(path_array), self.get_total_pathlength(np.array(path_array))

    def RandomSearch(self):
        '''
        For the random search we keep randomly choosing between the bottom and right adjacent cells
        until we reach the end
        :return:
        '''
        np.random.seed(self.seed)
        path_array = []
        self.iterations = 0
        # Set the last along with the maximum row and column count
        last_grid = [self.game_floor.shape[0] - 1, self.game_floor.shape[1] - 1]
        max_row = self.game_floor.shape[0] - 1
        max_col = self.game_floor.shape[1] - 1

        # Initialise the first cell and iteration count
        row = col = 0
        i = 0
        current_cell = [0, 0]

        # Let's start our search until we reach the end
        while last_grid != current_cell and i < 25:
            self.iterations += 1
            if (row + 1) > max_row:
                # We're at the bottom of the grid so we can only move right
                col += 1
                # print ("Move right")
            elif (col + 1) > max_col:
                # We're at the right of the grid so can only move down
                row += 1
                # print("Move down")
            elif np.random.randint(0, 1) == 0:
                # print("Move down")
                row += 1
            else:
                # print("Move right")
                col += 1

            current_cell = [row, col]
            path_array.append(current_cell)
            # print ("Current cell is {0}".format(current_cell))
        return self.iterations, np.array(path_array), self.get_total_pathlength(np.array(path_array))

    def DijkstrasSearch(self):
        '''
        We create 2 additional arrays wih the same shape as our
        game_floow array to help us implement the search algorithm
        The arrays are as follows:
        visted_array:   this tracks whether each grid in the game floor has been visited
        path_array:     this tracks the shortest path from the corresponding game floor grid to
                        the destination grid
        '''


        all_nodes = []
        for cur_row in range(0, self.max_row + 1):
            for cur_col in range(0, self.max_col + 1):
                nd = Node(cur_row,cur_col)
                all_nodes.append(nd)


        self.iterations = 0
        # create an N by M array with all cells, except the start cell, are set to "U" to represent unvisited
        visited_array = np.full((self.max_row + 1, self.max_col + 1), "U")

        # Set the start cell to "C" to signify it is the current cell
        visited_array[0, 0] = "C"

        # create an N by M array to track the shortest length
        # of each cell to the destination cell. All cells, except the start cell, are initialised to infinity
        path_array = np.full((self.max_row + 1, self.max_col + 1), float("inf"))

        # Set the start cell to 0 as the shortest length to itself is zero
        path_array[0, 0] = 0

        # We start moving to the neighbours of the current cell and
        # setting the minimum distance to the start cell

        for cur_row in range(0, self.max_row + 1):
            self.iterations += 1
            for cur_col in range(0, self.max_col + 1):
                self.iterations += 1
                # The current cell is denoted by [x,y]
                # Let's set the minimum distance of each adjacent cell from the
                # start cell
                cur_cell = np.array([[cur_row, cur_col]])
                visited_array[cur_row, cur_col] = "C"
                # If we are on any outer edge of the game_floor, ensure
                # that we do not look at adjacent cells that do not exist
                if cur_cell[0, 0] == 0:
                    rowoffsets = np.array([[1, 0]])
                elif cur_cell[0, 0] == self.max_row:
                    rowoffsets = np.array([[-1, 0]])
                else:
                    rowoffsets = np.array([[-1, 0], [1, 0]])

                if cur_cell[0, 1] == 0:
                    coloffsets = np.array([[0, 1]])
                elif cur_cell[0, 1] == self.max_col:
                    coloffsets = np.array([[0, -1]])
                else:
                    coloffsets = np.array([[0, -1], [0, 1]])

                offsets = np.append(rowoffsets, coloffsets, axis=0)

                for i in offsets:
                    self.iterations += 1
                    adjcell = cur_cell + i
                    if visited_array[adjcell[0, 0], adjcell[0, 1]] == "U":
                        path_len = self.game_floor[adjcell[0, 0], adjcell[0, 1]] + path_array[
                            cur_cell[0, 0], cur_cell[0, 1]]
                        if path_len < path_array[adjcell[0, 0], adjcell[0, 1]]:
                            path_array[adjcell[0, 0], adjcell[0, 1]] = int(path_len)
                            #The current cell is the closest to the adjacent cell so update its nearest neighbour
                            node = find_node(all_nodes,int(adjcell[0, 0]),int(adjcell[0, 1]) )
                            nearest_neighbour = find_node(all_nodes,int(cur_cell[0, 0]),int(cur_cell[0, 1]))
                            node.set_neighbour(nearest_neighbour)

                # set the cell to visited as we have checked all of it's neighbours
                visited_array[cur_cell[0, 0], cur_cell[0, 1]] = "V"

        #Get the shortest path
        shortest_path = np.array([[self.max_row,self.max_col]])
        last_node = find_node(all_nodes,self.max_row,self.max_col)
        nearest_neighbour = last_node.nearest_neighbour

        while nearest_neighbour != None:
            step = np.array([[nearest_neighbour.row,nearest_neighbour.col]])
            shortest_path = np.insert(shortest_path,0,step,0)
            nearest_neighbour = nearest_neighbour.nearest_neighbour

        shortest_path = np.delete(shortest_path,0,0)
        '''
        print("Game Floor: \n", self.game_floor)
        print("Visited Array: \n", visited_array)
        print("Path: \n",path_array.astype(int))
        '''
        return self.iterations, shortest_path, self.get_total_pathlength(shortest_path)

    def get_total_pathlength(self,path):
        total_length =  0
        for i in path:
            total_length += self.game_floor[i[0],i[1]]
        return total_length




















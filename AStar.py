import copy
from collections import deque, defaultdict
import math
import heapq
import random


class AStar:

    # Initializers for the class AStar that initializes the variables
    # with default values
    def __init__(self):
        self.grid = None
        self.openCells = None

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.distanceMatrix = None

    # This function takes a grid and converts it to tuple. In python,
    # 2-d array is unhashable because its mutable. Therefore,
    # we first convert it to immutable tuple and then hash it to
    # store in the visited set.
    def convertToTuple(self, grid):
        lst = []

        for i in grid:
            lst.append(tuple(i))

        return hash(tuple(lst))

    # This function takes a grid as an input and moves all the probabilities
    # in the grip in the upward direction by one value
    def moveUp(self, grid):

        if len(grid) <= 1:
            return copy.deepcopy(grid)

        rows = len(grid)
        cols = len(grid[0])

        newGrid = [[0 for i in range(cols)] for j in range(rows)]

        n = len(grid)

        for i in range(n - 2, -1, -1):
            for j in range(len(grid[0])):

                # If the current cell is not the cell and
                # the cell below it is not a wall,
                # we have a value below to send up.
                # We take the value at the cell below and
                # store it in the current cell in the
                # new grid
                if grid[i][j] != -1:
                    if grid[i + 1][j] != -1:
                        newGrid[i][j] = grid[i + 1][j]

                # If there is a wall, then the value at the cell below
                # needs to remain at the same place. Therefore, we add the
                # old value in the below cell to the same index in the new grid
                if grid[i][j] == -1:
                    if grid[i + 1][j] != -1:
                        newGrid[i + 1][j] += grid[i + 1][j]
                    newGrid[i][j] = -1

        # This is the base case of the grid. We initialize the first row
        # with the walls if there are any and the same with the last row.
        # The value at the first row remains at the same place. So we add the value
        # to the new grid at the first row.
        for j in range(len(grid[0])):
            if grid[0][j] != -1:
                if grid[1][j] != -1:
                    newGrid[0][j] += grid[0][j]
                else:
                    newGrid[0][j] = grid[0][j]
            else:
                newGrid[0][j] = -1

            if grid[-1][j] == -1:
                newGrid[-1][j] = -1
        return newGrid

    # This function is similar to the move up function described above.
    # This basically does the same operations in the opposite manner.
    def moveDown(self, grid):

        if len(grid) <= 1:
            return copy.deepcopy(grid)
        rows = len(grid)
        cols = len(grid[0])

        newGrid = [[0 for i in range(cols)] for j in range(rows)]

        n = len(grid)

        for i in range(1, n):
            for j in range(len(grid[0])):
                if grid[i][j] != -1:
                    if grid[i - 1][j] != -1:
                        newGrid[i][j] = grid[i - 1][j]

                if grid[i][j] == -1:
                    if grid[i - 1][j] != -1:
                        newGrid[i - 1][j] += grid[i - 1][j]
                    newGrid[i][j] = -1

        for j in range(len(grid[0])):
            if grid[-1][j] != -1:
                if grid[-2][j] != -1:
                    newGrid[-1][j] += grid[-1][j]
                else:
                    newGrid[-1][j] = grid[-1][j]
            else:
                newGrid[-1][j] = -1

            if grid[0][j] == -1:
                newGrid[0][j] = -1
        return newGrid

    # This function is similar to move up but moves all the values
    # to the right. All the cases mentioned above lie same but the row and column
    # iterations are changed to suit the right direction.
    def moveRight(self, grid):

        if len(grid[0]) <= 1:
            return copy.deepcopy(grid)

        rows = len(grid)
        cols = len(grid[0])

        newGrid = [[0 for i in range(cols)] for j in range(rows)]

        n = len(grid)

        for i in range(1, cols):
            for j in range(len(grid)):
                if grid[j][i] != -1:
                    if grid[j][i - 1] != -1:
                        newGrid[j][i] = grid[j][i - 1]

                if grid[j][i] == -1:
                    if grid[j][i - 1] != -1:
                        newGrid[j][i - 1] += grid[j][i - 1]
                    newGrid[j][i] = -1

        for j in range(len(grid)):
            if grid[j][-1] != -1:
                if grid[j][-2] != -1:
                    newGrid[j][-1] += grid[j][-1]
                else:
                    newGrid[j][-1] = grid[j][-1]
            else:
                newGrid[j][-1] = -1

            if grid[j][0] == -1:
                newGrid[j][0] = -1
        return newGrid

    # This function is similar to move right but instead moves to the left.
    def moveLeft(self, grid):

        if len(grid[0]) <= 1:
            return copy.deepcopy(grid)

        rows = len(grid)
        cols = len(grid[0])

        newGrid = [[0 for i in range(cols)] for j in range(rows)]

        n = len(grid)

        for i in range(cols - 2, -1, -1):
            for j in range(rows):
                if grid[j][i] != -1:
                    if grid[j][i + 1] != -1:
                        newGrid[j][i] = grid[j][i + 1]

                if grid[j][i] == -1:
                    if grid[j][i + 1] != -1:
                        newGrid[j][i + 1] += grid[j][i + 1]
                    newGrid[j][i] = -1

        for j in range(len(grid)):
            if grid[j][0] != -1:
                if grid[j][1] != -1:
                    newGrid[j][0] += grid[j][0]
                else:
                    newGrid[j][0] = grid[j][0]
            else:
                newGrid[j][0] = -1

            if grid[j][-1] == -1:
                newGrid[j][-1] = -1
        return newGrid

    # This is a helper function that takes in a grid and
    # returns the sum of all non-zero values in the grid.
    # This function has been used to debug the move operations.
    def printSum(self, grid):
        s = 0
        for i in grid:
            for j in i:
                if j != -1:
                    s += j
        return s

    # This function reads the text file and loads it into the
    # memory in the form of an array
    def load_grid(self):
        self.grid = []
        self.openCells = 0
        with open("input1.txt", "r") as f:
            reader = f.readlines()
            for r in reader:
                row = []
                for c in r:
                    if c == "_":
                        row.append(0)
                        self.openCells += 1
                    elif c == "X":
                        row.append(-1)

                self.grid.append(row)

    # This function is a helper function that iterates through the list.
    # If there is any open cell, it calculates the probability by 1 / number of open cells
    # and fills that cell with this value.
    def fill_empty_grid(self):

        n = len(self.grid)
        # print(self.grid)

        for i in range(n):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != -1:
                    self.grid[i][j] = 1 / self.openCells

    # This function is used to check if we have reached the final state
    # It iterates through the grid and checks if there is only 1 non-zero value.
    # The probability of 1 has not been checked to better scale it and make it work
    # for non-probabilistic values as well. But if there is only 1 non-zero value,
    # It returns a True else returns a False.
    def check_grid(self, grid):
        m = len(grid)
        n = len(grid[0])

        k = False
        for i in range(m):
            for j in range(n):
                if k and grid[i][j] > 0:
                    return False

                elif grid[i][j] > 0:
                    k = True

        return True

    # This function is a helper function that takes
    # an action or a list of actions to be performed on the given grid.
    # It performs the single or a batch of actions on the grid and returns
    # the new grid after performing the actions.
    def performAction(self, grid, action=[]):
        if type(action) == list:
            newGrid = copy.deepcopy(grid)
            for act in action:
                newGrid = self.performAction(newGrid, act)

            return newGrid
        else:
            if action == "UP":
                newGrid = self.moveUp(grid)
            elif action == "DOWN":
                newGrid = self.moveDown(grid)
            elif action == "LEFT":
                newGrid = self.moveLeft(grid)
            elif action == "RIGHT":
                newGrid = self.moveRight(grid)
            else:
                newGrid = copy.deepcopy(grid)

            return newGrid

    # This is a helper function that returns the number of non-zero values
    # in the grid. This has been used to debug the algorithms.
    def getNonZeroCount(self, grid):
        ctr = 0
        for i in grid:
            for j in i:
                if j > 0:
                    ctr += 1

        return ctr

    # This is a helper function that takes a grid
    # and prnts in a fancy understandable way
    def printGrid(self, grid):
        for row in grid:
            print("\t".join(map(str, row)))

    # This is the first function that gets called when we
    # wish to compute the solution.
    def compute(self):

        # If the grid is None, it loads the grid
        # and fills the grid with the non-empty probability
        if self.grid == None:
            self.load_grid()
            self.fill_empty_grid()

        # It now calls the astar method to compute
        # the answer to the grid.
        answer = self.astar()

        return answer

    # This function calculates the BFS given a destination.
    # The main idea is that the distance value source - destination
    # and destination - source is constant. Therefore,
    # it computes the distance values from all sources to the destination

    # This function is used to compute allPathsBFS that is a dynamic programming
    # method that initializes a 4-dimensional array
    # and stores the distances from all possible sources to all
    # possible destinations.
    def bfs(self, destination, grid):

        queue = deque()

        queue.append((destination, 0))

        visited = set()
        visited.add(destination)

        rows = [-1, 1, 0, 0]
        cols = [0, 0, -1, 1]

        # This variable stores the distances from all possible sources
        # to the given destination.
        path = [[-1 for i in range(len(grid[0]))] for j in range(len(grid))]

        while queue:
            elem = queue.popleft()
            row = elem[0][0]
            col = elem[0][1]
            for i in range(4):
                newRow = row + rows[i]
                newCol = col + cols[i]

                if (
                    0 <= newRow < len(grid)
                    and 0 <= newCol < len(grid[0])
                    and (newRow, newCol) not in visited
                    and grid[newRow][newCol] != -1
                ):
                    visited.add((newRow, newCol))
                    path[newRow][newCol] = elem[1] + 1
                    queue.append(((newRow, newCol), elem[1] + 1))

        return path

    # This is a dynamic programming method that has a 4-dimensional array
    # that computes the BFS for all possible destinations and
    # stores it in the 4-d matrix.
    # We use this function to get the distances in O(1) time.
    def allPathsBFS(self, grid):
        movi = [
            [
                [[0 for i in range(len(grid[0]))] for j in range(len(grid))]
                for k in range(len(grid[0]))
            ]
            for l in range(len(grid))
        ]

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                path = self.bfs((i, j), grid)
                for k in range(len(path)):
                    for l in range(len(path[0])):
                        movi[i][j][k][l] = path[k][l]

        self.distanceMatrix = movi

    # Given a source and destination, this function checks the distanceMatrix
    # for distance between a source to a destination and returns it in O(1) time.
    def computeDistance(self, p1, p2):

        if not self.distanceMatrix:
            self.allPathsBFS(self.grid)

        return self.distanceMatrix[p2[0]][p2[1]][p1[0]][p1[1]]

    # This function computes the maximum distance between two non-zero probability
    # nodes.
    def computeMaxDistance(self, grid):
        maxDist = 0
        n = len(grid)
        m = len(grid[0])

        n = len(grid)
        for i in range(n):
            for j in range(m):
                d = 0
                if grid[i][j] > 0:
                    for k in range(n):
                        for l in range(m):
                            if grid[k][l] > 0:
                                d = max(d, self.computeDistance((i, j), (k, l)))

                maxDist = max(maxDist, d)

        return maxDist

    # This function computes the heuristic.
    # the heuristic is a product of maximum distance that signifies it needs
    # to be converged. The number of non-zero probability nodes that
    # signifies that these values should be minimized and
    # the length of actions taken till now that also needs to
    # be as low as possible.
    # For two solutions that has a same distance and
    # same non-zero count, the value that has
    # the smaller length is preferred.
    def heuristic(self, grid, actions):

        return (
            (40 * self.computeMaxDistance(grid))
            * (8 * self.getNonZeroCount(grid))
            * len(actions)
        )

    # This is an A* algorithm that takes a heap and
    # at each step, tries to compute the heuristic by taking
    # 3 possible actions it can take. It avoids
    # the opposite of previous action at each time step.
    def astar(self):

        heap = []

        grid = copy.deepcopy(self.grid)

        h = self.heuristic(grid, [])

        heap.append((h, self.getNonZeroCount(grid), [], grid))

        visited = set()

        visited.add(hash(self.convertToTuple(grid)))

        answer = []
        ctr = 0
        while heap:
            elem = heapq.heappop(heap)
            # print(elem)
            ctr += 1

            if self.check_grid(elem[-1]):
                answer.append(elem[-2])
                self.printGrid(elem[-1])
                continue
                # return elem[1]

            if answer and ctr > 1000:
                # print(ctr)
                break

            actions = elem[-2]
            grid = elem[-1]
            heu = elem[0]

            for act in self.actions:

                # If the previous action is Left, then we avoid taking
                # a right. This helps in converging it better.
                if actions and actions[-1] == "LEFT" and act == "RIGHT":
                    continue

                # If the previous action is right, we avoid left
                if actions and actions[-1] == "RIGHT" and act == "LEFT":
                    continue

                # If the previous action is up, we avoid going  down.
                if actions and actions[-1] == "UP" and act == "DOWN":
                    continue

                # If the previous action is down, we avoid going up.
                if actions and actions[-1] == "DOWN" and act == "UP":
                    continue

                newGrid = self.performAction(grid, act)

                newHeu = self.heuristic(newGrid, actions + [act])

                # If the new state is not visited, then we add it to the heap.
                if hash(self.convertToTuple(newGrid)) not in visited:
                    visited.add(hash(self.convertToTuple(newGrid)))
                    heapq.heappush(
                        heap,
                        (
                            newHeu,
                            self.getNonZeroCount(newGrid),
                            actions[:] + [act],
                            newGrid,
                        ),
                    )

        return answer


if __name__ == "__main__":

    astar = AStar()

    answer = astar.compute()

    a = math.inf
    for i in answer:
        # print(len(i))
        a = min(a, len(i))

    for i in answer:
        if len(i) == a:
            print(i)
            print(len(i))

    # print(a)

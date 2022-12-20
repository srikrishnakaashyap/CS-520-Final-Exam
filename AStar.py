import copy
from collections import deque, defaultdict
import math
import heapq
import random


class AStar:
    def __init__(self):
        self.grid = None
        self.openCells = None

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def convertToTuple(self, grid):
        lst = []

        for i in grid:
            lst.append(tuple(i))

        return hash(tuple(lst))

    def moveUp(self, grid):

        if len(grid) <= 1:
            return copy.deepcopy(grid)

        rows = len(grid)
        cols = len(grid[0])

        newGrid = [[0 for i in range(cols)] for j in range(rows)]

        n = len(grid)

        for i in range(n - 2, -1, -1):
            for j in range(len(grid[0])):
                if grid[i][j] != -1:
                    if grid[i + 1][j] != -1:
                        newGrid[i][j] = grid[i + 1][j]

                if grid[i][j] == -1:
                    if grid[i + 1][j] != -1:
                        newGrid[i + 1][j] += grid[i + 1][j]
                    newGrid[i][j] = -1

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

    def printSum(self, grid):
        s = 0
        for i in grid:
            for j in i:
                if j != -1:
                    s += j
        return s

    def load_grid(self):
        # self.grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0, 0]]
        # self.openCells = 19
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

    def fill_empty_grid(self):

        n = len(self.grid)
        # print(self.grid)

        for i in range(n):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != -1:
                    self.grid[i][j] = 1 / self.openCells

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

    def printNonZeroCount(self, grid):
        ctr = 0
        for i in grid:
            for j in i:
                if j > 0:
                    ctr += 1

        return ctr

    def printGrid(self, grid):
        for row in grid:
            print("\t".join(map(str, row)))

    def compute(self):
        if self.grid == None:
            self.load_grid()
            self.fill_empty_grid()

        # print(self.grid)
        answer = self.astar()

        # self.grid = [
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1],
        #     [-1, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0, 1, -1],
        #     [-1, -1, -1, -1, 0, 0, 0, 0, 1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        # ]

        print(len(answer))
        return answer

    def computeDistance(self, p1, p2):
        return abs(p2[1] - p1[1]) + abs(p2[0] - p1[0])

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

    def heuristic(self, grid, actions):

        return (50 * self.computeMaxDistance(grid)) * (5 * self.printNonZeroCount(grid))

    def astar(self):

        heap = []

        grid = copy.deepcopy(self.grid)

        h = self.heuristic(grid, [])

        heap.append((h, [], grid))

        visited = set()

        visited.add(hash(self.convertToTuple(grid)))
        while heap:
            elem = heapq.heappop(heap)
            # print(elem)

            if self.check_grid(elem[2]):
                return elem[1]

            actions = elem[1]
            grid = elem[2]
            heu = elem[0]

            for act in self.actions:
                if actions and actions[-1] == "LEFT" and act == "RIGHT":
                    continue
                if actions and actions[-1] == "RIGHT" and act == "LEFT":
                    continue

                if actions and actions[-1] == "UP" and act == "DOWN":
                    continue

                if actions and actions[-1] == "DOWN" and act == "UP":
                    continue

                newGrid = self.performAction(grid, act)

                newHeu = self.heuristic(newGrid, actions + [act])

                if hash(self.convertToTuple(newGrid)) not in visited:
                    visited.add(hash(self.convertToTuple(newGrid)))
                    heapq.heappush(heap, (newHeu, actions[:] + [act], newGrid))

        return []


if __name__ == "__main__":

    astar = AStar()

    print(astar.compute())

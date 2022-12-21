from AStar import AStar
import math
import copy
import random


class LargestSequence:
    def __init__(self):
        self.grid = None
        self.openCells = None

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.distanceMatrix = None
        self.astar = AStar()

    # This is a helper function that takes a grid
    # and prnts in a fancy understandable way
    def printGrid(self, grid):
        for row in grid:
            print("\t".join(map(str, row)))

    def getOpenCount(self, grid):

        ctr = 0
        for i in grid:
            for j in i:
                if j == 0:
                    ctr += 1

        return ctr

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

    # This function computes the three bounded nodes
    # and tries to remove them thereby reducing the convergance points.
    def findLargestSequence(self):
        grid = self.grid
        self.printGrid(grid)
        threeBounded = []

        threeBoundedNodes = []
        rows = [-1, 1, 0, 0]
        cols = [0, 0, -1, 1]

        # This loop iterates through the grid
        # and finds all the three way convergence points
        for i in range(len(grid)):
            for j in range(len(grid[0])):

                ctr = 0

                for k in range(4):
                    newRow = i + rows[k]
                    newCol = j + cols[k]

                    if newRow < 0 or newRow >= len(grid) and 0 <= newCol < len(grid[0]):
                        ctr += 1
                    elif newCol < 0 or newCol >= len(grid) and 0 <= newRow < len(grid):
                        ctr += 1
                    else:
                        if (
                            0 <= newRow < len(grid)
                            and 0 <= newCol < len(grid[0])
                            and grid[newRow][newCol] == -1
                        ):
                            threeBoundedNodes.append((newRow, newCol))
                            ctr += 1

                if ctr == 3:
                    threeBounded.append((i, j))

        maxLen = 0
        maxSequence = []

        newGrid = copy.deepcopy(grid)
        while threeBoundedNodes:
            elem = random.choice(threeBoundedNodes)
            # print(k)
            threeBoundedNodes.remove(elem)
            # print(elem)

            # newGrid = copy.deepcopy(grid)
            newGrid[elem[0]][elem[1]] = 0
            self.astar.grid = newGrid
            self.astar.openCells = self.getOpenCount(newGrid)
            self.astar.fill_empty_grid()

            answer = self.astar.compute()

            a = math.inf
            for i in answer:
                # print(len(i))
                a = min(a, len(i))

            for i in answer:
                if len(i) == a:
                    print(i)
                    print(len(i))
                    if len(i) > maxLen:
                        maxSequence = i

        return maxSequence

    def compute(self):

        if self.grid is None:
            self.load_grid()

        newGrid = self.findLargestSequence()


if __name__ == "__main__":
    ls = LargestSequence()

    ls.compute()

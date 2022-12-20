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
        self.distanceMatrix = None

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

    def getNonZeroCount(self, grid):
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

        # print(len(answer))
        return answer

    def bfs(self, destination, grid):

        queue = deque()

        queue.append((destination, 0))

        visited = set()
        visited.add(destination)

        rows = [-1, 1, 0, 0]
        cols = [0, 0, -1, 1]
        path = [[-1 for i in range(len(grid[0]))] for j in range(len(grid))]
        directions = ["DOWN", "UP", "RIGHT", "LEFT"]

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

    def fourMatrix(self, grid):
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

    def computeDistance(self, p1, p2):

        if not self.distanceMatrix:
            self.fourMatrix(self.grid)

        return self.distanceMatrix[p2[0]][p2[1]][p1[0]][p1[1]]

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

        return (
            (40 * self.computeMaxDistance(grid))
            * (8 * self.getNonZeroCount(grid))
            * len(actions)
        )

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
                continue
                # return elem[1]

            if answer and ctr > 1000:
                print(ctr)
                break

            actions = elem[-2]
            grid = elem[-1]
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

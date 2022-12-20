import copy
from collections import deque, defaultdict
import math
import heapq
import random


class FinalExam:
    def __init__(self):
        self.grid = None
        self.openCells = None

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def convertToTuple(self, grid):
        lst = []

        for i in grid:
            lst.append(tuple(i))

        return hash(tuple(lst))

    def leftDist(self, grid):

        dist = math.inf

        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):
            d = 0
            encountered = False
            rowDist = math.inf
            for j in range(cols):
                if not encountered and grid[i][j] > 0:
                    encountered = True

                if encountered and grid[i][j] != -1:
                    d += 1

                if encountered and grid[i][j] == -1:
                    if d not in [0, 1]:
                        rowDist = min(rowDist, d - 1)
                    d = 0
                    encountered = False

            if d > 1:
                rowDist = min(rowDist, d)
            dist = min(dist, rowDist)
        return dist

    def rightDist(self, grid):

        dist = math.inf

        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):
            rowDist = math.inf
            d = 0
            encountered = False
            for j in range(cols - 1, -1, -1):

                if not encountered and grid[i][j] > 0:
                    encountered = True

                if encountered and grid[i][j] != -1:
                    d += 1

                if encountered and grid[i][j] == -1:
                    if d not in [0, 1]:
                        rowDist = min(rowDist, d - 1)
                    encountered = False
                    d = 0
            if d > 1:
                rowDist = min(rowDist, d)
            dist = min(dist, rowDist)

        return dist

    def topDist(self, grid):

        dist = math.inf

        rows = len(grid)
        cols = len(grid[0])

        for i in range(cols):
            d = 0
            encountered = False
            colDist = math.inf
            for j in range(rows):
                if not encountered and grid[j][i] > 0:
                    encountered = True

                if encountered and grid[j][i] != -1:
                    d += 1

                if encountered and grid[j][i] == -1:
                    if d not in [0, 1]:
                        colDist = min(colDist, d - 1)
                    encountered = False
                    d = 0
            if d > 1:
                colDist = min(colDist, d)

            # print(i, colDist)
            dist = min(dist, colDist)

        return dist

    def bottomDist(self, grid):

        dist = math.inf

        rows = len(grid)
        cols = len(grid[0])

        for i in range(cols - 1, -1, -1):
            d = 0
            encountered = False
            colDist = math.inf

            for j in range(rows - 1, -1, -1):
                if not encountered and grid[j][i] > 0:
                    encountered = True

                if encountered and grid[j][i] != -1:
                    d += 1

                if encountered and grid[j][i] == -1:
                    if d not in [0, 1]:
                        colDist = min(colDist, d - 1)
                    encountered = False
                    d = 0
            if d > 1:
                colDist = min(colDist, d)

            dist = min(dist, colDist)

        return dist

    def moveUp(self, grid):

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
        with open("input4.txt", "r") as f:
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

    def minimum_number_of_steps(self):

        if self.check_grid(self.grid):
            return []

        queue = deque()

        visited = set()
        queue.append(([], copy.deepcopy(self.grid)))
        answer = []

        visited.add(self.convertToTuple(self.grid))

        ctr = 0

        found = False
        while queue:

            n = len(queue)

            if found:
                break

            print(n)
            for i in range(n):
                elem = queue.popleft()
                ctr += 1
                # print(ctr, len(elem[0]))

                if self.check_grid(elem[1]):
                    # print(elem[0])
                    answer.append(elem[0])
                    found = True
                    continue
                for action in self.actions:
                    if action == "UP":
                        # print("action", action)
                        newGrid = self.moveUp(elem[1])
                        act = copy.deepcopy(elem[0])
                        act.append(action)

                        if self.convertToTuple(newGrid) not in visited:
                            queue.append((act, newGrid))
                            visited.add(self.convertToTuple(newGrid))
                    if action == "DOWN":
                        newGrid = self.moveDown(elem[1])
                        act = copy.deepcopy(elem[0])
                        act.append(action)
                        if self.convertToTuple(newGrid) not in visited:
                            queue.append((act, newGrid))
                            visited.add(self.convertToTuple(newGrid))

                    if action == "LEFT":
                        newGrid = self.moveLeft(elem[1])
                        act = copy.deepcopy(elem[0])
                        act.append(action)
                        # print(act)
                        if self.convertToTuple(newGrid) not in visited:
                            queue.append((act, newGrid))
                            visited.add(self.convertToTuple(newGrid))

                    if action == "RIGHT":
                        newGrid = self.moveRight(elem[1])
                        act = copy.deepcopy(elem[0])
                        act.append(action)
                        if self.convertToTuple(newGrid) not in visited:
                            queue.append((act, newGrid))
                            visited.add(self.convertToTuple(newGrid))

        # print(answer)
        return answer

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
                    path[newRow][newCol] = (directions[i], elem[1] + 1)
                    queue.append(((newRow, newCol), elem[1] + 1))

        return path

    def changeGame(self, grid, actions):

        while True:
            if self.check_grid(grid):
                return actions

            pathMap = {}

            maxDist = 0

            source = None
            destination = None

            for i in range(len(grid)):
                for j in range(len(grid[0])):

                    if grid[i][j] > 0:

                        path = self.bfs((i, j), grid)

                        pathMap[(i, j)] = path

                        for k in range(len(grid)):
                            for l in range(len(grid[0])):

                                if k != i and l != j:
                                    if grid[k][l] > 0:
                                        if path[k][l][1] > maxDist:
                                            maxDist = path[k][l][1]

    def heuristic(self, grid, action):

        newGrid = self.performAction(grid, action)

        oldCount = self.printNonZeroCount(grid)
        newCount = self.printNonZeroCount(newGrid)

        heuristic = None

        if action == "LEFT":
            heuristic = self.rightDist(self.grid) / (oldCount - newCount + 1)
        elif action == "RIGHT":
            heuristic = self.leftDist(self.grid) / (oldCount - newCount + 1)
        elif action == "UP":
            heuristic = self.bottomDist(self.grid) / (oldCount - newCount + 1)
        else:
            heuristic = self.topDist(self.grid) / (oldCount - newCount + 1)

        return heuristic

    def greedy(self):
        actions = []

        grid = copy.deepcopy(self.grid)

        ctr = 0

        visited = defaultdict(list)

        action = None
        while not self.check_grid(grid):

            ctr += 1
            dmap = defaultdict(list)

            if action != "LEFT":
                leftDist = self.leftDist(grid)
                dmap[leftDist].append("RIGHT")

            if action != "RIGHT":
                rightDist = self.rightDist(grid)
                dmap[rightDist].append("LEFT")

            if action != "UP":
                topDist = self.topDist(grid)
                dmap[topDist].append("DOWN")

            if action != "DOWN":
                downDist = self.bottomDist(grid)
                dmap[downDist].append("UP")

            dmap.pop(math.inf, None)

            minD = max(dmap.keys())

            action = dmap[minD][0]

            # newGrid = copy.deepcopy(grid)

            act = [action] * (minD - 1)

            newGrid = copy.deepcopy(grid)

            for a in act:

                newGrid = self.performAction(grid, a)
                if newGrid == grid:
                    print("SAME")
                else:
                    actions.append(action)

            grid = newGrid

            h = hash(self.convertToTuple(grid))
            if h not in visited:
                visited[h] = actions[:]
            else:
                print(visited[h], len(visited[h]))
                print("VISITED")
                print(self.printNonZeroCount(grid))

        return actions

    def optimal(self):
        actions = []

        grid = copy.deepcopy(self.grid)

        action = None

        heap = []
        while self.check_grid(grid) is False:

            hmap = defaultdict(list)

            if action != "LEFT":
                hmap[self.heuristic(grid, "RIGHT")].append("RIGHT")

            if action != "RIGHT":
                hmap[self.heuristic(grid, "LEFT")].append("LEFT")

            if action != "UP":
                hmap[self.heuristic(grid, "DOWN")].append("DOWN")

            if action != "DOWN":
                hmap[self.heuristic(grid, "UP")].append("UP")

            minHeuristic = min(hmap.keys())
            action = random.choice(hmap[minHeuristic])

            newGrid = self.performAction(grid, action)

            if newGrid == grid:
                print("SAME")
                print(self.bottomDist(grid))
                print(self.printNonZeroCount(grid))
            else:
                actions.append(action)

            grid = copy.deepcopy(newGrid)

        return actions

    def printGrid(self):
        # print("INSIDE PRINT")
        for row in self.grid:

            print("\t".join(map(str, row)))

    def printGrid(self, grid):
        for row in grid:
            print("\t".join(map(str, row)))

    def compute(self):
        if self.grid == None:
            self.load_grid()
            self.fill_empty_grid()

        # print(self.grid)
        answer = self.optimal()

        # self.grid = [
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1],
        #     [-1, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0, 1, -1],
        #     [-1, -1, -1, -1, 0, 0, 0, 0, 1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        # ]

        print(len(answer))
        return answer

        topDist = self.bottomDist(self.grid)
        print(topDist)
        # directions = self.bfs((1, 5), (2, 1), self.grid)
        # self.printGrid(directions)


if __name__ == "__main__":

    fe = FinalExam()

    print(fe.compute())

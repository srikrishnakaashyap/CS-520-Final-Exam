import copy
from collections import deque, defaultdict
import math
import heapq


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

        dist = len(grid)

        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):

            d = 0
            encountered = False
            for j in range(cols):
                if not encountered and grid[i][j] > 0:
                    encountered = True

                if encountered and grid[i][j] > 0:
                    d += 1

                if encountered and grid[i][j] == -1:
                    break

            if d not in [0, 1]:
                dist = min(dist, d)
            # print(dist, i)

        return dist

    def rightDist(self, grid):

        dist = len(grid)

        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows - 1, -1, -1):

            d = 0
            encountered = False
            for j in range(cols):
                if not encountered and grid[i][j] > 0:
                    encountered = True

                if encountered and grid[i][j] > 0:
                    d += 1

                if encountered and grid[i][j] == -1:
                    break

            if d not in [0, 1]:
                dist = min(dist, d)

        return dist

    def topDist(self, grid):

        dist = len(grid[0])

        rows = len(grid)
        cols = len(grid[0])

        for i in range(cols):
            d = 0
            encountered = False

            for j in range(rows):
                if not encountered and grid[j][i] > 0:
                    encountered = True

                if encountered and grid[j][i] > 0:
                    d += 1

                if encountered and grid[j][i] == -1:
                    break

            if d not in [0, 1]:
                dist = min(dist, d)

        return dist

    def bottomDist(self, grid):

        dist = len(grid[0])

        rows = len(grid)
        cols = len(grid[0])

        for i in range(cols):
            d = 0
            encountered = False

            for j in range(rows - 1, -1, -1):
                if not encountered and grid[j][i] > 0:
                    encountered = True

                if encountered and grid[j][i] > 0:
                    d += 1

                if encountered and grid[j][i] == -1:
                    break

            if d not in [0, 1]:
                dist = min(dist, d)

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
        with open("input3.txt", "r") as f:
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

    def dijkstras(self):
        if self.check_grid(self.grid):
            return []

        heap = []

        heap.append((0, [], copy.deepcopy(self.grid)))

        while heap:

            elem = heapq.heappop(heap)

            if self.check_grid(elem[2]):
                return elem[1]

            dmap = defaultdict(list)

            leftDist = self.leftDist(elem[2])
            dmap[leftDist].append("LEFT")

            rightDist = self.rightDist(elem[2])
            dmap[rightDist].append("RIGHT")

            topDist = self.topDist(elem[2])
            dmap[topDist].append("UP")

            downDist = self.bottomDist(elem[2])
            dmap[downDist].append("DOWN")

            minD = min(dmap.keys())

            for act in dmap[minD]:
                moves = copy.deepcopy(elem[1])
                actions = [act] * (minD - 1)

                newGrid = self.performAction(elem[2], actions)

                moves += actions
                heapq.heappush(heap, (len(moves), moves, newGrid))

    def printNonZeroCount(self, grid):
        ctr = 0
        for i in grid:
            for j in i:
                if j > 0:
                    ctr += 1

        return ctr

    def greedy(self):
        actions = []

        grid = copy.deepcopy(self.grid)

        ctr = 0

        while not self.check_grid(grid):

            ctr += 1
            if ctr == 80:
                print("TEST")

            dmap = defaultdict(list)

            leftDist = self.leftDist(grid)
            dmap[leftDist].append("LEFT")

            rightDist = self.rightDist(grid)
            dmap[rightDist].append("RIGHT")

            topDist = self.topDist(grid)
            dmap[topDist].append("UP")

            downDist = self.bottomDist(grid)
            dmap[downDist].append("DOWN")

            minD = min(dmap.keys())

            action = dmap[minD][0]

            acts = [action] * (minD)

            newGrid = self.performAction(grid, acts)

            if newGrid == grid:
                print(self.printNonZeroCount(grid))
                print("TEST")

            grid = newGrid

            actions += acts

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

        answer = self.greedy()
        return answer
        # l = self.moveUp(self.grid)

        # self.printGrid(l)

        # d = self.moveDown(l)
        # l = self.moveUp(d)
        # self.printGrid(l)

        # self.printGrid(self.grid)
        # l = self.moveDown(self.grid)
        # # print(l)
        # print("----------------")
        # self.printGrid(l)
        # m = self.moveDown(l)
        # print("----------------")
        # self.printGrid(m)
        # print("----------------")
        # m = self.moveRight(m)
        # self.printGrid(m)
        # print("----------------")
        # n = self.moveRight(m)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveRight(m)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveDown(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveRight(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveDown(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveDown(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveDown(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveDown(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # n = self.moveLeft(n)
        # self.printGrid(n)
        # print("----------------")
        # m = self.moveDown(l)m = self.moveDown(l)m = self.moveDown(l)

        # print(m)
        # n = self.moveLeft(m)
        # print(n)

        # n = self.moveUp(n)
        # print(n)
        # self.grid = n
        # print(self.check_grid(n))
        # print(self.printSum(n))

        # self.grid = [
        #     [0.5, 0.5, 0.5, 0.5, -1],
        #     [0.5, 0.5, 0.5, -1, 0.5],
        #     [0.5, 0.5, -1, 0.5, 0.5],
        #     [0.5, -1, 0.5, 0.5, 0.5],
        #     [0.5, 0.5, 0.5, 0.5, 0.5],
        # ]

        # self.grid = [[0.5, 0.5, 0.5], [0.5, -1, 0.5], [0.5, 0.5, 0.5]]

        # print("LENGTH", len(self.grid))
        # print("LENGTH 0", len(self.grid[0]))

        # self.printGrid()

        # answer = self.minimum_number_of_steps()
        # # # print(answer)
        # for ans in answer:
        #     print(ans)

        # leftDist = self.leftDist(self.grid)
        # rightDist = self.rightDist(self.grid)
        # topDist = self.topDist(self.grid)
        # bottomDist = self.bottomDist(self.grid)

        # print(leftDist, rightDist, topDist, bottomDist)

    # print(answer)


if __name__ == "__main__":

    fe = FinalExam()

    print(fe.compute())

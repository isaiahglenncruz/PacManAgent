from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

from pacai.core import distance
import heapq
"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    stack = Stack()
    visited = []
    stack.push((problem.startingState(), [], 0))

    while not stack.isEmpty():
        state, path, cost = stack.pop()
        if problem.isGoal(state):
            return path
        if state in visited:
            continue

        visited.append(state)
        successors = problem.successorStates(state)
        for successor in successors:
            stack.push((successor[0], path + [successor[1]], successor[2]))
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    stack = Queue()
    visited = []
    stack.push((problem.startingState(), [], 0))

    while not stack.isEmpty():
        state, path, cost = stack.pop()
        if problem.isGoal(state):
            return path
        if state in visited:
            continue

        visited.append(state)
        successors = problem.successorStates(state)
        for successor in successors:
            stack.push((successor[0], path + [successor[1]], successor[2]))
    return None

# https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs
def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    initial = problem.startingState()
    initial_3 = (initial, [], 0)
    q = PriorityQueue()
    q.push(initial_3, initial_3[2])

    expanded = set()
    path = None
    in_frontier = 0

    while(1):
        if len(q.heap) == 0:
            return None
        pac_node = q.pop()
        pac_state = pac_node[0]

        if problem.isGoal(pac_state) is True:
            path = pac_node[1]
            return path

        expanded.add(pac_state)

        for successor in problem.successorStates(pac_state):
            list_moves = pac_node[1]
            list_moves.append(successor[1])
            item = (successor[0], list_moves, successor[2])

            for priority, node in q.heap:
                if (node[0] == item[0]):
                    in_frontier = 1
                    break
            if (item[0] not in expanded) and (in_frontier == 0):
                q.push(item, item[2])

            elif in_frontier == 1:
                for (priority, node) in q.heap:
                    if node[0] == item[0] and priority < item[2]:
                        q.remove((priority, node))
                        q.push(item, item[2])
                        break
            in_frontier = 0

def null_heuristic(problem, state):
    return 1

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    open_list = PriorityQueue()
    closed_list = []

    initial = problem.startingState()
    open_list.push((initial, [], heuristic(initial, problem)), heuristic(initial, problem))

    while not open_list.isEmpty():
        x = open_list.pop()
        state = x[0]
        path = x[1]
        if problem.isGoal(state):
            return path

        closed_list.append(state)

        successors = problem.successorStates(state)
        for successor in successors:
            pair = successor[0]

            h = heuristic(pair, problem)
            if type(state[0]) == tuple:
                g = distance.manhattan(state[0], pair[0])
            if type(state[0]) == int:
                g = distance.manhattan(state, pair)

            f = g + h

            item = (successor[0], path + [successor[1]], f)

            open_better = 0
            closed_better = 0

            for i in open_list.heap:
                obj = i[1]
                temp_state = obj[0]
                if temp_state == successor[0]:
                    temp_h = heuristic(temp_state, problem)
                    temp_g = 0
                    if type(state[0]) == tuple:
                        temp_g = distance.manhattan(state[0], temp_state[0])
                    if type(state[0]) == int:
                        temp_g = distance.manhattan(state, temp_state)
                    temp_f = temp_g + temp_h
                    if temp_f <= f:
                        open_better = 1
            if open_better == 1:
                continue

            for i in closed_list:
                temp_state = i
                if temp_state == successor[0]:
                    temp_h = heuristic(temp_state, problem)
                    temp_g = 0
                    if type(state[0]) == tuple:
                        temp_g = distance.manhattan(state[0], temp_state[0])
                    if type(state[0]) == int:
                        temp_g = distance.manhattan(state, temp_state)
                    temp_f = temp_g + temp_h
                    if temp_f <= f:
                        closed_better = 1
            if closed_better == 1:
                continue

            for i in open_list.heap:
                obj = i[1]
                temp_state = obj[0]
                if temp_state == successor[0]:
                    open_list.heap.remove(i)
            heapq.heapify(open_list.heap)
            for i in closed_list:
                temp_state = i
                if temp_state == successor[0]:
                    closed_list.remove(i)

            open_list.push(item, heuristic(pair, problem))

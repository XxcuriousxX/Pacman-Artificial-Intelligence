# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    # util.raiseNotDefined()
    stack = Stack()  #stack variable works like the fringe to manage state
    visited =set()     #we keep track of visited states
    track = []      #list to keep path for every state from start

    node = problem.getStartState()
    #we check if start state is Goal State
    if (problem.isGoalState(node)):
        return []

    stack.push([node,track])

    while (True):

        if stack.isEmpty():
            return []

        #we pop position and path of current state and add the node to visited set
        node,path = stack.pop()
        visited.add(node)

        #if we have reach our goal we return the path
        if problem.isGoalState(node):
            return path

        #expanding current state and getting the successors
        successor = problem.getSuccessors(node)

        #for every successor we check if it is visited   (algorithm is implemented as in the lectures)
        #if it is not we push it into the stack with the updated path = cost of path until we reach node + the cost of path to the successor
        for x in successor:
            if x[0] not in visited:
                newPath = path + [x[1]]
                stack.push((x[0],newPath))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import Queue
    #BFS is very similar to DFS implimentation.We have here queue instead of stack.
    queue = Queue()
    visited = set() #we keep track of visited states
    track = []  #list to keep path for every state from start

    node = problem.getStartState()

    #we check if start state is Goal State
    if (problem.isGoalState(node)):
        return []

    queue.push([node,track])

    while True:

        if queue.isEmpty():
            return []

        #we pop position and path of current state and add the node to visited set
        node = queue.pop()
        visited.add(node[0])

        #if we have reach our goal we return the path
        #node[0] = node
        #node[1] = path
        if problem.isGoalState(node[0]):
            return node[1]


        successor = problem.getSuccessors(node[0])

        #for every successor we check if it is visited
        #if it is not in visited and not in queue as well (algorithm is implemented as in the lectures)
        #we push it into the stack with the updated path = cost of path until we reach node + the cost of path to the successor
        for x in successor:
            if x[0] not in visited and x[0] not in (state[0] for state in queue.list):

                newPath = node[1] + [x[1]]
                queue.push((x[0],newPath))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import PriorityQueue

    frontier = PriorityQueue() #we know that ucs uses priority queue so we implement in this way

    visited = set() #we keep track of visited states
    path = []    #list to keep path for every state from start

    node = problem.getStartState()
    #we check if start state is Goal State
    if (problem.isGoalState(node)):
        return []

    #we start from the first node - path list is an empty list at this point
    frontier.push([node,path],problem.getCostOfActions(path))

    while (True):

        #unable to find solution
        if (frontier.isEmpty()):
            return []

        #we pop from the priority queue the node and the path of current state
        node,path = frontier.pop()

        #we check if we have reached our GoalState and we return the solution
        if problem.isGoalState(node):
            return path

        #if we havent reached goal state we add node to the visited set and we get its successors
        visited.add(node)
        succ = problem.getSuccessors(node)

        #for every successor we check if it is in visited list  (algorithm is implemented as in the lectures)
        #and in the queue as well and we act accordingly
        for item in succ:
            #successor is not in visited set and not in the queue as well
            if item[0] not in visited and item[0] not in (state[2][0] for state in frontier.heap):
                #we update the path with path from parent to child
                updated_path = list(path)
                updated_path.append(item[1])
                #we push the the child in the queue with the updated path and priority equal to the updated_path cost
                priority = problem.getCostOfActions(updated_path)
                frontier.push([item[0],updated_path],priority)

            #successor is not in visited set and it is in the queue
            elif item[0] not in visited and item[0] in (state[2][0] for state in frontier.heap):
                #we update the path with path from parent to child
                updated_path = list(path)
                updated_path.append(item[1])
                for each in frontier.heap:
                    #we check if the child is in the priority queue and if the updated_path path has less priority
                    #so we have to update it in the queue. The path and the priority as well.
                    if each[2][0] == item[0]:
                        if each[0] > problem.getCostOfActions(updated_path):
                            each[2][1] = updated_path
                            cost = problem.getCostOfActions(updated_path)
                            frontier.update([item[0],updated_path],cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    fringe = PriorityQueue()   #fringe holding states ready to be expanded form --> ((state,action,cost),priority)
    path = []   #list to keep path for every state from start to state
    visited = set() #Set to keep track of visited states
    node = problem.getStartState()


    #we push as priority the f(n) = g(n) + h(n)
    fringe.push([node,path],problem.getCostOfActions(path)+heuristic(node,problem))

    #we check if start state is Goal State
    if (problem.isGoalState(node)):
        return []


    while not fringe.isEmpty():

        current = fringe.pop()
        #we check if we have visited current node and if not we add it in visited set
        if(current[0] not in visited):
            visited.add(current[0])

            #we check if we have reached our goal
            if problem.isGoalState(current[0]):
                return current[1]

            successors = problem.getSuccessors(current[0])
            #for each child we compute our entry which we will push to the queue
            for childs in successors:
                if childs[0] not in visited:
                    #we append in our path the path from parent to child
                    update_path=list(current[1])
                    update_path.append(childs[1])
                    #we push into the queue the child with the updated path and the f = g + h as the priority
                    fringe.push([childs[0],update_path],problem.getCostOfActions(update_path)+heuristic(childs[0],problem))

    return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

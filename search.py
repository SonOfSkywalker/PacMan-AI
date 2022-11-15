# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#


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

    startPos = problem.getStartState() # Get the starting position of the pacman
    pred = {} # We will use this dictionnary to backtrack the path to return

    if (problem.isGoalState(problem.getStartState())): # start state == goal ?
        return [] # No move necessary as state = goal

    frontier = util.Stack() # LIFO (Last In First Out), needed to a deep exploration

    for state, action, cost in problem.getSuccessors(startPos): # We go through every successors of the starting node

        pred[(state, action, cost)] = None # Thanks to Mrs. ABDEDDAIM
        frontier.push((state, action, cost)) # We add our node to the frontier, ie the nodes to process

    explored = [startPos] # We only add the origin position to the explored positions

    while not(frontier.isEmpty()): # While we have some nodes to process

        node = frontier.pop() # Choose the deepest node in the frontier (as it is a LIFO)

        for successorNode in problem.getSuccessors(node[0]): # We go through the successors

            if not(successorNode[0] in explored): # If you haven't already explored it

                pred[successorNode] = node # Thanks to Mrs. ABDEDDAIM

                frontier.push(successorNode) # Add to frontier
                explored.append(successorNode[0]) # Thanks to Mrs. ABDEDDAIM

                if (problem.isGoalState(successorNode[0])): # Successeur == goal ?

                    path = [] # We start to build our path
                    currentNode = successorNode # We start from the goal

                    while (currentNode != None): # While we do not reach the starting node
                        path.insert(0,currentNode[1]) # We insert the action at the beginning of our list
                        currentNode = pred[currentNode] # The next node to process is the predecessor of the current one
                    return path # Return path to goal

    return [] # If we reach this return, no food was found

def breadthFirstSearch(problem): # Similar to BFS, we only change LIFO to FIFO

        # See comments on DFS for further Information

        startPos = problem.getStartState()
        pred = {}

        if (problem.isGoalState(problem.getStartState())):
            return []

        frontier = util.Queue() # FIFO (First In First Out), needed to a shallow exploration

        for state, action, cost in problem.getSuccessors(startPos):

            pred[(state, action, cost)] = None
            frontier.push((state, action, cost))

        explored = [startPos]

        while not(frontier.isEmpty()):

            node = frontier.pop() # Choose the shallowest node in the frontier (as it is a FIFO)

            for successorNode in problem.getSuccessors(node[0]):

                if not(successorNode[0] in explored):

                    pred[successorNode] = node

                    frontier.push(successorNode)
                    explored.append(successorNode[0])

                    if (problem.isGoalState(successorNode[0])):

                        path = []
                        currentNode = successorNode

                        while (currentNode != None):
                            path.insert(0,currentNode[1])
                            currentNode = pred[currentNode]
                        return path

        return []


def uniformCostSearch(problem): # similar to BFS except that we use a PriorityQueue and update the costs

    # See comments on DFS for further Information

    startPos = problem.getStartState()
    pred = {}

    if (problem.isGoalState(problem.getStartState())):
        return []

    frontier = util.PriorityQueue() # We use a PriorityQueue to sort the queue by the costs of each node

    for state, action, cost in problem.getSuccessors(startPos):

        pred[(state, action, cost)] = None
        frontier.push((state, action, cost), cost) # We append our node and the cost linked with it

    explored = [startPos]

    while not(frontier.isEmpty()):

        node = frontier.pop() # Choose the shallowest node in the frontier (as it is a FIFO)

        for successorNode in problem.getSuccessors(node[0]):

            if not(successorNode[0] in explored):

                pred[successorNode] = node

                frontier.update(successorNode, node[2] + successorNode[2]) # We update our PQueue using the sum of the node and its predecessor
                explored.append(successorNode[0])

                if (problem.isGoalState(successorNode[0])):

                    path = []
                    currentNode = successorNode

                    while (currentNode != None):
                        path.insert(0,currentNode[1])
                        currentNode = pred[currentNode]
                    return path

    return []




def nullHeuristic(state, problem=None):
    return 0

def manhattanHeuristic(state, problem=None):
    return manhattanDistance( problem.goal, state )





def aStarSearch(problem, heuristic=nullHeuristic): # Same as UCS except for heuristic

        # See comments on UCS for further Information

        startPos = problem.getStartState()
        pred = {}

        if (problem.isGoalState(problem.getStartState())):
            return []

        frontier = util.PriorityQueue()

        for state, action, cost in problem.getSuccessors(startPos):

            pred[(state, action, cost)] = None
            frontier.push((state, action, cost), cost)


        explored = [startPos]

        while not(frontier.isEmpty()):

            node = frontier.pop()

            for successorNode in problem.getSuccessors(node[0]):

                if not(successorNode[0] in explored):

                    pred[successorNode] = node

                    frontier.update(successorNode, node[2] + successorNode[2] + heuristic(successorNode[0], problem)) # We update our PQueue using the sum of the node and its predecessor + the heuristic value of this state
                    explored.append(successorNode[0])

                    if (problem.isGoalState(successorNode[0])):

                        path = []
                        currentNode = successorNode

                        while (currentNode != None):


                            path.insert(0,currentNode[1])
                            currentNode = pred[currentNode]
                        return path

        return []

def FoodaStarSearch(problem, heuristic=nullHeuristic): # Same as UCS except for heuristic

        # See comments on UCS for further Information

        startPos = problem.getStartState()
        pred = {}

        if (problem.isGoalState(problem.getStartState())):
            return []

        frontier = util.PriorityQueue()

        for state, action, cost in problem.getSuccessors(startPos):

            pred[(state, action, cost)] = None
            frontier.push((state, [action], cost), cost)


        explored = [startPos]

        while not(frontier.isEmpty()):

            node = frontier.pop()

            for successorNode in problem.getSuccessors(node[0]):

                if not(successorNode[0] in explored):

                    pred[successorNode] = node

                    frontier.update( (successorNode[0], node[1] + [successorNode[1]], successorNode[2]), node[2] + successorNode[2] + heuristic(successorNode[0], problem)) # We update our PQueue using the sum of the node and its predecessor + the heuristic value of this state
                    explored.append(successorNode[0])

                    if (problem.isGoalState(successorNode[0])):
                        return node[1] + [successorNode[1]]

        return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

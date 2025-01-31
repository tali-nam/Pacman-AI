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

    visited = set()
    frontier = util.Stack()
    previousNode = dict()
    route = []

    start = problem.getStartState()
    if(problem.isGoalState(start)):
        return start
    frontier.push(start)

    while(not frontier.isEmpty()):
    
          if(frontier.list[-1] not in visited):
            curr = frontier.pop()
            visited.add(curr)
          else:
              frontier.pop()
          
          if(problem.isGoalState(curr)):
              while(curr != start):
                  directions = previousNode[curr][1]
                  route.insert(0,directions)
                  curr = previousNode[curr][0]
              return route
                  
          successors = problem.getSuccessors(curr)
          for i in successors:
              if(i[0] not in visited):
                  frontier.push(i[0]) 
                  previousNode[i[0]] = (curr, i[1])
                
#should add nodes to fringe multiple times 
def breadthFirstSearch(problem):
    #initialize data structures 

    visited = []
    frontier = util.Queue()


    startState = problem.getStartState()
    start = (startState,[],0)
    frontier.push(start)

    while(not frontier.isEmpty()):
        state, action, cost = frontier.pop()
    
        if(state not in visited):
            visited.append(state)
            

            if(problem.isGoalState(state)):
                    return action
            
            successors = problem.getSuccessors(state)
            for (newState,newAction,newCost) in successors:
                
                updatedActions = action+[newAction]
                updatedCost= cost+newCost
                newNode = (newState, updatedActions, updatedCost)

                if(newState not in visited):
                    frontier.push(newNode)
    else:
        return action

def uniformCostSearch(problem):
        #initialize data structures 

    visited = set()
    frontier = util.PriorityQueue()


    startState = problem.getStartState()
    start = (startState,[],0)
    frontier.push(start,0)

    while(not frontier.isEmpty()):
        state, action, cost = frontier.pop()
    
        if(state not in visited):
            visited.add(state)
            

            if(problem.isGoalState(state)):
                    return action
            
            successors = problem.getSuccessors(state)
            for (newState,newAction,newCost) in successors:
                
                updatedActions = action+[newAction]
                updatedCost= cost+newCost
                newNode = (newState, updatedActions, updatedCost)

                if(newState not in visited):
                    frontier.push(newNode, updatedCost)
    else:
        return action
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """Search the node of least total cost first."""

    ''' setting up data structures '''

    visited = []
    frontier = util.PriorityQueue()

    startState = problem.getStartState()
    start = (startState,[],0)
    frontier.push(start,0)


    while(not frontier.isEmpty()):
        state, action, cost = frontier.pop()

        if(state not in visited):
            visited.append(state)
            


            if(problem.isGoalState(state)):
                    return action
            
            successors = problem.getSuccessors(state)
            for (newState,newAction,newCost) in successors:
                
                updatedActions = action+[newAction]
                updatedCost= cost+newCost
                newNode = (newState, updatedActions, updatedCost)


                heuristicCost = updatedCost + heuristic(newState, problem)


                if(newState not in visited):
                    frontier.push(newNode, heuristicCost)
    else:
        return action

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

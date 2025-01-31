# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        minFoodDistance = float ('inf')
        visitedFood = []
        rows, cols = newFood.width, newFood.height
        frontierFood = util.Queue()
        
        "ghost penalty"
        ghostPenalty  = float('-inf')
        

        for i in range(0,len(newGhostPos)):
            if newScaredTimes[i] == 0 and newGhostPos[i] == newPos:
                return ghostPenalty
        
        if action == 'Stop':
            return -9999
       
        x, y = newPos

        if (oldFood[x][y]):
            return 0
    
        frontierFood.push(newPos)
        
        while(not frontierFood.isEmpty()):
            thisX,thisY = frontierFood.pop()
            if((thisX,thisY) not in visitedFood):
                visitedFood.append((thisX,thisY))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
                newX, newY = thisX + dx, thisY + dy
                if(0<=newX<rows and 0<=newY<cols):
                    if(newFood[newX][newY]):
                        minFoodDistance = manhattanDistance(newPos, (newX, newY))
                        return -minFoodDistance
            
                    if((newX,newY) not in visitedFood):
                        frontierFood.push((newX,newY))

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        --> what does these do?

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        Ghosts minimize, pacman maximizes
        """
        "*** YOUR CODE HERE ***"
        "Returns the max of the mins"
        "Max plays first"
        actions = gameState.getLegalActions(0)
        maxV = float('-inf')
        for a in actions:
            "Successor state"
            nextState = gameState.generateSuccessor(0,a)
            thisV = self.minValue(nextState,0,1)
            if(thisV>maxV):
                maxV = thisV
                maxA = a
        return maxA
    
    def minValue(self, gameState, currDepth, currAgent):
        "Retun the min of the maxs "
        "If terminal -- if win/lose/or reach the end of tree depth"
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        numberOfGhosts = gameState.getNumAgents() -1
        minV  = float("inf")
        "Need all the ghosts to move to evaluate next state"
        "Each ghost needs to move optimally"
        "For each ghost"
        actions = gameState.getLegalActions(currAgent)
        for a in actions:
            nextState = gameState.generateSuccessor(currAgent,a)
            "If the last ghost moves"
            if(currAgent == numberOfGhosts):
                v = self.maxValue(nextState,currDepth + 1,0)

            else:
                v = self.minValue(nextState, currDepth, currAgent+1)
            minV = min(minV, v) 
        '''print("minimum V", minV)'''
        return minV

    def maxValue(self, gameState, currDepth, currAgent):
        "Max of the mins"
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        maxV = float("-inf")
        actions = gameState.getLegalActions(currAgent)
        for a in actions:
            "return the max of the mins"
            nextState = gameState.generateSuccessor(currAgent,a)
            maxV = max(maxV, self.minValue(nextState, currDepth,1))
        '''print("maximumV", maxV)'''
        return maxV


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        "best guaranteed option for max"
        alpha = float("-inf")
        "best guaranteed option for min"
        beta = float("inf")
        "best value for action"
        maxValue = float("-inf")

        "for every legal action"
        for a in actions:
            "evaluate the successor state"
            nextState = gameState.generateSuccessor(0,a)
            "calculate v value for state"
            thisV = self.minValue(nextState, alpha, beta,0,1)
            if(thisV>maxValue):
                maxValue = thisV
                maxA = a
                alpha = max(alpha, maxValue)
        return maxA
    
    def maxValue(self, state, alpha, beta, currDepth):
        "Maximizing agent --> Take the max of the mins"
        "Return minimax action while pruning"

        "If terminal state (win/lose/cutoff depth)"
        if state.isWin() or state.isLose() or currDepth == self.depth:
            return self.evaluationFunction(state)
        "minimum v"
        v = float("-inf")
        "Legal actions for Pacman (maximizing) agent"
        actions = state.getLegalActions(0)
        for a in actions:
            thisState = state.generateSuccessor(0,a)
            "Mins of the maxes, what is the minimizing agent going to do"
            thisV = self.minValue(thisState, alpha, beta, currDepth, 1)
            "Maximum of v and thisV"
            if(v<thisV):
                v = thisV
            "beta is the best guaranteed option for min"
            if v>beta:
                "don't need to check tree because min will never choose option larger than beta"
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, state, alpha, beta, currDepth, currAgent):
        "If terminal state (win/lose/cutoff depth)"
        if state.isWin() or state.isLose() or currDepth == self.depth:
            return self.evaluationFunction(state)
        v = float("inf")
        numberOfGhosts = state.getNumAgents() -1
        actions = state.getLegalActions(currAgent)
        for a in actions:
            nextState = state.generateSuccessor(currAgent,a)
            "If the last ghost moves"
            if(currAgent == numberOfGhosts):
                v = min(v,self.maxValue(nextState, alpha, beta, currDepth + 1))
            else:
                v = min(v,self.minValue(nextState, alpha, beta, currDepth, currAgent+1))
            if (v<alpha):
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        maxV = float('-inf')
        for a in actions:
            "Successor state"
            nextState = gameState.generateSuccessor(0,a)
            thisV = self.minValue(nextState,0,1)
            if(thisV>=maxV):
                maxV = thisV
                maxA = a
        return maxA
        
        util.raiseNotDefined()

    "Ghosts choose action at random"
    def minValue(self, gameState, currDepth, currAgent):
        "Retun the min of the maxs "
        "If terminal -- if win/lose/or reach the end of tree depth"
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        numberOfGhosts = gameState.getNumAgents() -1
        minV  = float("inf")
        "Need all the ghosts to move to evaluate next state"
        "Each ghost needs to move optimally"
        "For each ghost"
        actions = gameState.getLegalActions(currAgent)
        totalV = 0;
        for a in actions:
            nextState = gameState.generateSuccessor(currAgent,a)
            "If the last ghost moves"
            if(currAgent == numberOfGhosts):
                v = self.maxValue(nextState, currDepth + 1, 0)

            else:
                v = self.minValue(nextState, currDepth, currAgent+1)
            totalV += v
        '''print("minimum V", minV)'''
        return totalV/len(actions)
    
    "Max value should continue picking the best option"

    def maxValue(self, gameState, currDepth, currAgent):
        "Max of the mins"
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        maxV = float("-inf")
        actions = gameState.getLegalActions(currAgent)
        for a in actions:
            "return the max of the mins"
            nextState = gameState.generateSuccessor(currAgent,a)
            maxV = max(maxV, self.minValue(nextState, currDepth,1))
        '''print("maximumV", maxV)'''
        return maxV

    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 

    I wanted to incorporate as much information about the game state while also making my evaluation function simple (so that it was also efficient)!
    So, I used ghost information, food, magic pellets, and the score to evaluate the state. I also added different scores so that it was simple and efficient.
    """
    "*** YOUR CODE HERE ***"

    "Winning is the absolute best case scenerio"

    if(currentGameState.isWin()):
        return float('inf')
    
    "Losing is the absoltue worst case scenerio"
    
    if(currentGameState.isLose()):
        return float('-inf')
    
    "Higher scores are better, lower scores are worst"
    pacman = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    ghostStates= currentGameState.getGhostStates()
   
    "Time"
    score = currentGameState.getScore()*2

    evaluation = score

    "How close is the closest ghost"

    "SMALLER VALUES ARE BAD for scary ghosts"
    "SMALLER VALUES ARE GOOD for scared ghosts"

    minScaryGhostPos = float("inf")
    minScaredGhostPos = float("inf")
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    scaredGhostBonus = 0

    for i in range (0, len(ghosts)):
        ghostDistance = util.manhattanDistance(pacman,ghosts[i])
        if(newScaredTimes[i] > 0):
            if(ghostDistance<minScaredGhostPos):
                minScaredGhostPos = ghostDistance
        else:
            if(ghostDistance<minScaryGhostPos):
                minScaryGhostPos = ghostDistance
    
    if minScaryGhostPos == float("inf") :
        scaredGhostBonus = 60
        minScaryGhostPos = 0
    
    if minScaredGhostPos == float("inf"):
        minScaredGhostPos = 0
    

    evaluation = evaluation + minScaryGhostPos - minScaredGhostPos + scaredGhostBonus 
        
    "How far is the food"

    "SMALLER VALUES ARE GOOD"

    food = currentGameState.getFood()
    foodDistance = 0

    numFood = 0

    for x in range(0,food.width):
        for y in range(0,food.height):
            if(food[x][y]):
                foodDistance += util.manhattanDistance(pacman, (x,y))
                numFood += 1
    

    evaluation = evaluation - foodDistance

    "How far is the magic food"

    "SMALLER VALUES ARE GOOD"

    magicFood = currentGameState.getCapsules()
    magicDistance = 0
    numMagicFood = 0
    for m in magicFood:
        magicDistance += util.manhattanDistance(m, pacman)
    
    evaluation = evaluation - magicDistance

    return evaluation


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

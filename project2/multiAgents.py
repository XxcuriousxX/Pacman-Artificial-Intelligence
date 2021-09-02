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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        #predefined reward if pacman is in a food position
        infood_points = 5

        #calculating the manhattanDistance from current position to every position where we have food
        food = newFood.asList()
        food_manhattan = [manhattanDistance(newPos,each) for each in food]

        #if we are already in a food pellet we add in score a +5 reward
        if newPos in food:
            score += infood_points
        #if there is no food left we return score else we calculate the distance to the closest food
        if(len(food_manhattan)!=0):
            to_eat = min(food_manhattan)
        else:
            return score



        ghost_pos = successorGameState.getGhostPositions()
        ghost_manhattan = [manhattanDistance(newPos,ghost) for ghost in ghost_pos]
        ghost_near = min(ghost_manhattan)

        #if ghost is closer than 2 steps we try to reduce our score
        #in order to force our agent to avoid the ghost
        if ghost_near < 2:
            score -= 100



        #we return an estimate value, which is the reward to eat food divided by the distance reaching the food
        return score + 5 / to_eat



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
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState,self.depth,0)[1]
        # util.raiseNotDefined()
    def max_value(self,state,depth,index):
        #checking for terminal state
        #we return max value and move as well (in tuple).
        if depth == 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
        value = float("-inf")
        # for each legal action we generate the Successor and we get its min value
        for action in state.getLegalActions(index):
            successor = state.generateSuccessor(index,action)
            score = self.min_value(successor,depth,index+1)[0]
            # we keep track of max value and its move in current depth
            if score > value:
                value = score
                move = action
        return (value,move)


    def min_value(self,state,depth,index):
        # terminal state
        if depth == 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)

        value = float("inf")
        if index < state.getNumAgents()-1:  #ghost turn (min player)
            agent = index + 1 #we increase agent index next ghost turn.
            for action in state.getLegalActions(index): #for every legal action we generate the successor and keep track
            # of score (recursive call of min_value)
                successor = state.generateSuccessor(index,action)
                score = self.min_value(successor,depth,agent)[0]
                # we update move and value if we find a lower score.
                if score < value:
                    move = action
                    value = score
        else:
            agent = 0 #pacman turn (max player)
            # for each legal action we generate the Successor and compute the max value
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index,action)
                score = self.max_value(successor,depth-1,agent)[0]
                # we update move and value if we find a lower score.
                if score < value:
                    move = action
                    value = score
        return (value,move)






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.MaxValue(gameState,self.depth,0, float("-inf"), float("inf"))[1]

    def MaxValue(self,state,depth, agentIndex, alpha, beta):
        # terminal states
        if depth == 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)


        value = float("-inf")
        # for each legal action we generate successor and compute the minimizer value
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = self.MinValue(successor,depth,agentIndex+1, alpha, beta)[0]
            # if we compute a bigger value than the previous max
            # we update value and action
            if score > value:
                value = score
                maxAction = action
            # we prune the tree (alpha > beta)
            if value > beta:
                return (value,maxAction)
            # if we dont prune the tree we get update alpha with max (value - alpha)
            alpha = max(alpha,value)
        return (value,maxAction)

    def MinValue(self,state,depth,agentIndex, alpha, beta):
        # terminal states
        if depth == 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
        value = float("inf")
        # if we have min player's turn (ghost)
        if(agentIndex < state.getNumAgents()-1):
            # we have next players turn now (ghost or pacman) ,we increase index by 1.
            agent = agentIndex + 1
            # for each legal action of ghost we compute the minimizer for successor
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex,action)
                score = self.MinValue(successor,depth,agent,alpha,beta)[0]
                # we update value and action if we find a better one
                if score < value:
                    value = score
                    minAction = action
                # prune tree if we have beta less than alpha
                if value < alpha:
                    return (value,minAction)
                # we update beta
                beta = min(beta,value)
        else:   # we have max player's turn (pacman)
            agent = 0   # pacman is always at index 0
            #for each legal action we compute minimizer score of the successor
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex,action)
                score = self.MaxValue(successor,depth-1,agent,alpha,beta)[0]
                # we update value and its action if we find a better one
                if score < value:
                    value = score
                    minAction = action
                # if alpha is bigger than beta we prune the tree
                if value < alpha:
                    return (value,minAction)
                # we update beta value
                beta = min(beta,value)
        return (value,minAction)


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
        # util.raiseNotDefined()
        max_value = -float("inf")
        action  = None
        # we get the max value from all its successors
        for each_action in gameState.getLegalActions(0):
            score = self.expValue(gameState.generateSuccessor(0,each_action),1,0)
            if((score) > max_value): #we update max value and action if we find better one
                max_value = score
                action = each_action
        return action


    def maxValue(self,gameState,depth):
        # terminal states
        if((depth == self.depth) or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)

        # we return the max of expected values for each pacman successor's
        # if we get pacman's legal actions
        return max([self.expValue(gameState.generateSuccessor(0,each),1,depth) for each in gameState.getLegalActions(0)])

    def expValue(self,gameState,index,depth):
        # we store the amount of legal actions left for our ghost
        actions_len = len(gameState.getLegalActions(index))

        # if no legal actions left we return evaluation Function
        if (actions_len == 0):
            return self.evaluationFunction(gameState)
        # we check if we have more ghosts left and we act accordingly
        if (index < gameState.getNumAgents() -1):
            # we return the exp value which is the sum of exp values for each ghost successor
            # and for all the possible legal actions it can make divided by the amount of legal actions left so we get the stohastic value.
            return sum([self.expValue(gameState.generateSuccessor(index,each),index+1,depth) for each in gameState.getLegalActions(index)]) / float(actions_len)
        else:
            # if all ghost's have played we return the sum of max values for each successor
            # and for all the possible legal actions it can make divided by the amount of legal actions left so we get the stohastic value.
            return sum([self.maxValue(gameState.generateSuccessor(index,each),depth+1) for each in gameState.getLegalActions(index)]) / float(actions_len)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # we store pacman position,capsules,ghost,food positions
    capsules = currentGameState.getCapsules()
    position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    score = currentGameState.getScore()
    food_list = currentGameState.getFood().asList()
    # initializing our variables
    closestfood = 1
    closestghost = float("inf")
    closestcapsule = float("inf")

    # for each capsule we take the min distance between current capsule and pacman
    cdist = []
    for each in capsules:
        cdist.append(manhattanDistance(each,position))
    if len(cdist) > 0:
        closestcapsule = min(cdist)
    ###############################################################################

    # for each ghost we take the min distance between current ghost and pacman
    for each in ghosts:
        gdist = manhattanDistance(each,position)
        closestghost = min(closestghost,manhattanDistance(each,position))
        if gdist <= 1:
            closestfood = -float ("inf")
    ############################################################################


    # for each food we take the min distance between current food and pacman
    gfood =[]
    for each in food_list:
        gfood.append(manhattanDistance(each,position))
    if len(food_list) > 0:
        closestfood = min(gfood)
    ############################################################################

    # we return score based on weights for each of our features
    score = score
    score += 6.0 / closestfood
    score += - 4.5  * len(currentGameState.getCapsules())
    score += - 10 * len(food_list)
    score += - 3.5 * closestghost
    score += 8.0 / closestcapsule

    return score
better = betterEvaluationFunction

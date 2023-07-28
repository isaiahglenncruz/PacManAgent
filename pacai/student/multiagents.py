import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        limit = 999999999

        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        old_food_position_list = oldFood.asList()
        old_food_amount = len(old_food_position_list)
        if old_food_amount == 0:
            limit = 0
        else:
            for i in range(old_food_amount):
                check_dist = distance.manhattan(newPosition, old_food_position_list[i]) + 100000
                if check_dist < limit:
                    limit = check_dist
        score = 0 - limit

        ghost_states_amount = len(newGhostStates)
        for i in range(ghost_states_amount):
            ghost_position_tuple = successorGameState.getGhostPosition(i + 1)
            ghost_position = (int(ghost_position_tuple[0]), int(ghost_position_tuple[1]))
            check_dist = distance.manhattan(newPosition, ghost_position)
            if check_dist <= 1:
                score -= 999999999

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, agent_index, tree_depth):
        num_actions = len(gameState.getLegalActions(agent_index))
        if num_actions == 0 or tree_depth == self._treeDepth:
            return (self._evaluationFunction(gameState), "")
        if agent_index == 0:
            return self.max_value(gameState, agent_index, tree_depth)
        else:
            return self.min_value(gameState, agent_index, tree_depth)

    def max_value(self, gameState, agent_index, tree_depth):
        max_value, max_action = -999999999, ""
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_value = self.value(successor, agent_index, tree_depth + 1)[0]
            if curr_value > max_value:
                max_value = curr_value
                max_action = action
        return (max_value, max_action)

    def min_value(self, gameState, agent_index, tree_depth):
        min_value, min_action = -999999999, ""
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_value = self.value(successor, agent_index, tree_depth + 1)[0]
            if curr_value > min_value:
                min_value = curr_value
                min_action = action
        return (min_value, min_action)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.ab_search(gameState, 0, 0, -999999999, 999999999)[1]

    def ab_search(self, gameState, agent_index, tree_depth, alpha, beta):
        num_actions = len(gameState.getLegalActions(agent_index))
        if num_actions == 0 or tree_depth == self._treeDepth:
            return (self._evaluationFunction(gameState), "")
        if agent_index == 0:
            return self.max_value(gameState, agent_index, tree_depth, alpha, beta)
        else:
            return self.min_value(gameState, agent_index, tree_depth, alpha, beta)

    def max_value(self, gameState, agent_index, tree_depth, alpha, beta):
        max_value, max_action = -999999999, ""
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_value = self.ab_search(successor, agent_index, tree_depth + 1, alpha, beta)[0]
            if curr_value > max_value:
                max_value = curr_value
                max_action = action
            alpha = max(alpha, max_value)
            if max_value > beta:
                return (max_value, max_action)
        return (max_value, max_action)

    def min_value(self, gameState, agent_index, tree_depth, alpha, beta):
        min_value, min_action = 999999999, ""
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index)
            curr_value = self.ab_search(successor, agent_index, tree_depth + 1, alpha, beta)[0]
            if curr_value < min_value:
                min_value = curr_value
                min_action = action
            beta = min(beta, min_value)
            if min_value < alpha:
                return (min_value, min_action)
        return (min_value, min_action)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, agent_index, tree_depth):
        num_actions = len(gameState.getLegalActions(agent_index))
        if num_actions == 0 or tree_depth == self._treeDepth:
            return (self._evaluationFunction(gameState), "")
        if agent_index == 0:
            return self.max_value(gameState, agent_index, tree_depth)
        else:
            return self.exp_value(gameState, agent_index, tree_depth)

    def max_value(self, gameState, agent_index, tree_depth):
        max_value, max_action = -999999999, ""
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_value = self.value(successor, agent_index, tree_depth + 1)[0]
            if curr_value > max_value:
                max_value = curr_value
                max_action = action
        return (max_value, max_action)

    def exp_value(self, gameState, agent_index, tree_depth):
        exp_value, exp_action = 0, ""
        legal_actions = gameState.getLegalActions(agent_index)
        weight = float(1.0 / len(legal_actions))
        for action in legal_actions:
            successor = gameState.generateSuccessor(agent_index, action)
            exp_value += weight * self.value(successor, agent_index, tree_depth + 1)[0]
        return (exp_value, exp_action)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 999999999
    if currentGameState.isLose():
        return -999999999

    score = 0

    food = currentGameState.getFood().asList()
    food_count = len(food)
    score -= 5 * food_count

    position = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    ghost_states_amount = len(ghost_states)
    ghost_dist_list = []
    for i in range(ghost_states_amount):
        ghost_position_tuple = currentGameState.getGhostPosition(i + 1)
        ghost_position = (int(ghost_position_tuple[0]), int(ghost_position_tuple[1]))
        dist = distance.manhattan(position, ghost_position)
        ghost_dist_list.append(dist)
    ghost_dist_min = min(ghost_dist_list)
    if ghost_dist_min < 5:
        score -= pow(5 - ghost_dist_min, 5)

    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

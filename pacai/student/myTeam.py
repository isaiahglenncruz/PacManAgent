from curses.ascii import FF
from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.layout import Layout

import random

def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        TheeAgent(firstIndex),
        TheeAgent(secondIndex),
    ]

def isInTunnel(agent_index, width, gameState):
    red_tunnel = width - 2 # width = 32, tunnel = 30, wall = 31
    blue_tunnel = 1 # wall = 0, tunnel = 1
    agent_pos = gameState.getAgentPosition(agent_index)
    if agent_index == 0 or agent_index == 2:
        if float(agent_pos[0]) == float(blue_tunnel):
            return True
        else:
            return False
    elif agent_index == 1 or agent_index == 3:
        if float(agent_pos[0]) == float(red_tunnel):
            return True
        else:
            return False

class TheeAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
        self.spaces = []
        self.red_spaces = []
        self.blue_spaces = []
        self.actions = 0
        self.index = index
        self.board_width = 0
        self.board_half = 0

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.
        spaces = []
        walls = gameState.getWalls().asList(False)
        for position in walls:
            spaces.append(position)

        self.board_width = gameState._layout.width
        self.board_half = self.board_width / 2
        self.red_spaces = [i for i in spaces if i[0] < (self.board_half)]
        self.blue_spaces = [i for i  in spaces if i[0] >= (self.board_half)]
        print("width: ", self.board_width, "halfway point: ", self.board_half)

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """
        
        super().chooseAction(gameState)
        actions = gameState.getLegalActions(self.index)
        self.actions += 1
        action = random.choice(actions)
        # curr_agent_state = gameState.getAgentState(self.index)
        # curr_pos = curr_agent_state.getPosition()
        # if (self.index == 1 or self.index == 3):
        #     in_tunnel = isInTunnel(self.index, self.board_width, gameState)
        #     print("\nagent: ", self.index, "position: ", curr_pos, "action: ", action, "inTunnel: ", in_tunnel)
        #     print("blue team agent 0: ", isInTunnel(0, self.board_width, gameState))
        return action
    
    

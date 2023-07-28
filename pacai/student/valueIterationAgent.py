from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        for i in range(1):
            for state in mdp.getStates():
                self.values[state] = 0

        for i in range(self.iters):
            next_values = dict(self.values)
            for state in mdp.getStates():
                values = []
                if self.mdp.isTerminal(state):
                    values.append(0)
                else:
                    for action in mdp.getPossibleActions(state):
                        values.append(self.getQValue(state, action))
                next_values[state] = max(values)
            self.values = dict(next_values)

    def getQValue(self, state, action):
        sp = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0

        for state_prob in sp:
            reward = self.mdp.getReward(state, action, state_prob[0])
            q += state_prob[1] * (reward + (self.discountRate * self.values[state_prob[0]]))
        return q

    def getPolicy(self, state):
        actions = self.mdp.getPossibleActions(state)
        policy = None
        current_q_value = -999999999

        for action in actions:
            next_q_value = self.getQValue(state, action)
            if next_q_value > current_q_value:
                current_q_value = next_q_value
                policy = action
        return policy

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

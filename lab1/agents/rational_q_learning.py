from typing import override

import numpy as np

from lab1.agents.q_agent_base import QLearningAgentBase


class RationalQLearningAgent(QLearningAgentBase):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions
        )

    @override
    def update(self, state, action, reward, next_state_idx, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state_idx])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    @override
    def get_metrics(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

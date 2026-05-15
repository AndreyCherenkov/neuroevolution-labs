from abc import ABC
from typing import override

import numpy as np

from lab1.agents.base import BaseAgent


class QLearningAgentBase(BaseAgent, ABC):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 1.0,
    ):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    @override
    def act(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        return int(np.argmax(self.Q[state]))

    @override
    def update_epsilon(self, attenuation_coefficient: float = 0.997, min_epsilon: float = 0.01):
        self.epsilon = max(self.epsilon * attenuation_coefficient, min_epsilon)

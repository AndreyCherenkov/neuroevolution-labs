
import numpy as np

from lab1.agents.base import BaseAgent


class RationalQLearningAgent(BaseAgent):
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

    def act(self, state_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state_idx]).astype(int)

    def update(self, state, action, reward, next_state_idx, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state_idx])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

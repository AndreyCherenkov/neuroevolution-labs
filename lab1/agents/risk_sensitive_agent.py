import numpy as np

from lab1.agents.base import BaseAgent


class RiskSensitiveAgent(BaseAgent):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            eta: float = 2.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.num_actions = num_actions

        self.Q = np.zeros((num_states, num_actions))
        self.U = np.exp(-self.eta * self.Q)

    def act(self, state_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        return int(np.argmax(self.Q[state_idx]))

    def update(
            self,
            state_idx: int,
            action_idx: int,
            reward: float,
            next_state_idx: int,
            done: bool
    ):
        reward_term = np.exp(-self.eta * reward)
        if done:
            target_u = reward_term
        else:
            min_next_u = np.min(self.U[next_state_idx])
            target_u = reward_term * (min_next_u ** self.gamma)

        current_u = self.U[state_idx, action_idx]

        updated_u = (1 - self.alpha) * current_u + self.alpha * target_u

        updated_u = max(updated_u, 1e-12)  # todo read

        self.U[state_idx, action_idx] = updated_u

        self.Q[state_idx, action_idx] = -np.log(updated_u) / self.eta

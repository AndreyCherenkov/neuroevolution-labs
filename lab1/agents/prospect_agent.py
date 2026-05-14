import numpy as np

from lab1.agents.base import BaseAgent


class ProspectAgent(BaseAgent):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            alpha_p: float = 0.88,
            beta_p: float = 0.88,
            lambda_p: float = 2.35,
    ):
        self.Q = np.zeros([num_states, num_actions])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p
        self.num_actions = num_actions

    def act(self, state_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state, action, reward, next_state, done):
        subjective_reward = self.__value_function(reward)
        if done:
            target = subjective_reward
        else:
            target = subjective_reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error

    def __value_function(self, reward: float) -> float:
        if reward >= 0:
            return reward ** self.alpha_p
        return -self.lambda_p * ((-reward) ** self.beta_p)

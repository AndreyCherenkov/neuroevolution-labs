from typing import override

import numpy as np

from lab1.agents.q_agent_base import QLearningAgentBase


class ProspectAgent(QLearningAgentBase):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            alpha_p: float = 0.88,
            beta_p: float = 0.88,
            lambda_p: float = 2.35,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions
        )

        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p

    @override
    def update(self, state, action, reward, next_state, done):
        subjective_reward = self.__value_function(reward)
        if done:
            target = subjective_reward
        else:
            target = subjective_reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error

    @override
    def get_metrics(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "alpha_p": self.alpha_p,
            "beta_p": self.beta_p,
            "lambda_p": self.lambda_p,
        }

    def __value_function(self, reward: float) -> float:
        if reward >= 0:
            return reward ** self.alpha_p
        return -self.lambda_p * ((-reward) ** self.beta_p)

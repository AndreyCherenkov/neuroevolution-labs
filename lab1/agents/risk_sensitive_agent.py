from typing import override

import numpy as np

from lab1.agents.q_agent_base import QLearningAgentBase


class RiskSensitiveAgent(QLearningAgentBase):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            eta: float = 2.0,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions
        )

        self.eta = eta

        self.U = np.exp(-self.eta * self.Q)


    @override
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

    @override
    def get_metrics(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "eta": self.eta,
        }

from typing import Iterable

from lab1.agents.base import BaseAgent
from lab1.env.environment import Environment
from lab1.telemetry.episode_log import EpisodeLog


def train(
        env: Environment,
        agent: BaseAgent,
        episodes: int = 15000,
) -> Iterable[EpisodeLog]:
    logs = []
    for episode in range(episodes):
        state = env.reset()
        done = False

        summary_reward = 0
        is_collided = False

        while not done:
            state_idx = state.discretize().to_index()

            action_idx = agent.act(state_idx)
            next_state, reward, done, collided = env.step(action_idx)

            next_state_idx = next_state.discretize().to_index()

            agent.update(state_idx, action_idx, reward, next_state_idx, done)

            state = next_state

            summary_reward += reward
            if collided:
                is_collided = True

        agent.update_epsilon()

        logs.append(EpisodeLog(summary_reward, env.time, int(is_collided), state.battery))

    return logs
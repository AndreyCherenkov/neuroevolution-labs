from lab1.agents.q_agent_base import QLearningAgentBase
from lab1.env.environment import Environment
from lab1.telemetry.test_metric import TestMetric


def test_agents(
        env: Environment,
        agent: QLearningAgentBase,
        episodes: int = 300,
        epsilon: float = 0.005
) -> TestMetric:
    old_epsilon = agent.epsilon

    agent.epsilon = epsilon

    successes = 0
    success_episode_length = 0
    charge_cost = 0
    collisions = 0

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            state_idx = state.discretize().to_index()
            action_ids = agent.act(state_idx)

            next_state, reward, done, is_success, is_collided = env.step(action_ids)

            state = next_state

            successes += successes
            charge_cost += next_state.battery
            collisions += is_collided
            if is_success:
                success_episode_length += env.time

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return TestMetric(
        success_percentage=(successes / episodes) * 100,
        average_success_episode_length=success_episode_length / successes, # ZeroDivisionError
        average_charge_cost=charge_cost / episodes,
        average_episode_collision_percentage=(collisions / episodes) / 100
    )

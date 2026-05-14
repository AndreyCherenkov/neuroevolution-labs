from lab1.agents.rational_q_learning import BaseAgent
from lab1.env.environment import Environment


def train(
        env: Environment,
        agent: BaseAgent,
        episodes: int = 15000
):
    for episode in range(episodes):
        state = env.reset()
        done = False
        t = 0

        while not done:
            state_idx = state.discretize().to_index()

            action_idx = agent.act(state_idx)
            next_state, reward, done = env.step(action_idx, t)

            next_state_idx = next_state.discretize().to_index()

            agent.update(state_idx, action_idx, reward, next_state_idx, done)

            state = next_state
            t += 1

        if hasattr(agent, "epsilon"):
            agent.epsilon = max(agent.epsilon * 0.997, 0.01)

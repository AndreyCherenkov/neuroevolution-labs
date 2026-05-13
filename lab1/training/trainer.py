from lab1.agents.rational_q_learning import BaseAgent, RationalQLearningAgent
from lab1.env.agent_spawn import AgentSpawn
from lab1.env.environment import Environment
from lab1.env.obstacles import Obstacle
from lab1.env.recharge_station import RechargeStation
from lab1.env.wind import Wind


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

        if isinstance(agent, RationalQLearningAgent):
            agent.epsilon = max(agent.epsilon * 0.995, 0.01)


if __name__ == '__main__':
    en = Environment(
        (0, 24),
        (0, 12),
        RechargeStation(),
        AgentSpawn(),
        Wind(),
        obstacles=[
            Obstacle(10.0, 2.0, 11.5, 5.5),
            Obstacle(13.0, 7.0, 14.5, 10.0),
            Obstacle(16.5, 3.5, 18.0, 6.5),
            Obstacle(19.0, 9.0, 20.5, 11.5),
        ]
    )

    q_agent = RationalQLearningAgent(9408, 12)
    train(en, q_agent, 150)
    for x in q_agent.Q:
        print(x)

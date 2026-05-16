from typing import Callable, Iterable

import mlflow
from lab1.agents.base import BaseAgent
from lab1.agents.q_agent_base import QLearningAgentBase
from lab1.env.environment import Environment
from lab1.telemetry.episode_log import EpisodeLog
from lab1.telemetry.test_metric import TestMetric


def log_data(
        experiment_name: str,
        run_name: str,
        train_func: Callable[[Environment, BaseAgent, int], Iterable[EpisodeLog]],
        env: Environment,
        agent: BaseAgent,
        episodes: int = 15000
) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # todo define in config
    mlflow.set_experiment(experiment_name)

    print(f'Experiment name: {experiment_name}')
    print(f'Run name: {run_name}')

    with mlflow.start_run(run_name=run_name):
        for name, value in agent.get_metrics().items():
            mlflow.log_param(name, value)

        logs = train_func(env, agent, episodes)

        for index, log in enumerate(logs):
            mlflow.log_metrics({
                "reward": log.summary_reward,
                "energy_consumption": log.final_charge,
                "episode_length": log.episode_length,
                "collision": log.collision_number,
            }, step=index)


def compute_metrics(
        experiment_name: str,
        run_name: str,
        test_func: Callable[[Environment, QLearningAgentBase, int, float], TestMetric],  # todo refactor?
        env: Environment,
        agent: QLearningAgentBase,
        episodes: int = 300,
        epsilon: float = 0.005
) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    print(f'Experiment name: {experiment_name}')
    print(f'Run name: {run_name}')

    with mlflow.start_run(run_name=run_name):
        for name, value in agent.get_metrics().items():
            mlflow.log_param(name, value)

        metric = test_func(env, agent, episodes, epsilon)

        mlflow.log_metrics({
            "success_percentage": metric.success_percentage,
            "average_success_episode_length": metric.average_success_episode_length,
            "average_charge_cost": metric.average_charge_cost,
            "average_episode_collision_percentage": metric.average_episode_collision_percentage,
        })

from typing import Callable

import mlflow
from lab1.agents.base import BaseAgent
from lab1.env.environment import Environment
from lab1.telemetry.episode_log import EpisodeLog


def log_data(
        experiment_name: str,
        run_name: str,
        train_func: Callable[[Environment, BaseAgent, int], list[EpisodeLog]],
        env: Environment,
        agent: BaseAgent,
        episodes: int = 15000
) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    print(f'Experiment name: {experiment_name}')
    print(f'Run name: {run_name}')

    with mlflow.start_run(run_name=run_name):
        for name, value in agent.get_metrics().items():
            mlflow.log_metric(name, value)

        logs = train_func(env, agent, episodes)

        for index, log in enumerate(logs):
            mlflow.log_metrics({
                "reward": log.summary_reward,
                "energy_consumption": log.final_charge,
                "episode_length": log.episode_length,
                "collision": log.collision_number,
            }, step=index)

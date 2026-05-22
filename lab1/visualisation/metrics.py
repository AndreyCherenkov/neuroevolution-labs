from typing import Callable, Iterable, Any

import mlflow
from matplotlib import pyplot as plt

from lab1.agents.base import BaseAgent
from lab1.agents.q_agent_base import QLearningAgentBase
from lab1.env.environment import Environment
from lab1.telemetry.episode_log import EpisodeLog
from lab1.telemetry.test_metric import TestMetric
from lab1.visualisation.graphics import create_success_time_cdf_fig, \
    create_trajectories_fig, create_learning_curve_fig, create_final_charge_heatmap_fig
from lab1.visualisation.t_test import perform_pure_python_ttest


def log_data(
        experiment_name: str,
        run_name: str,
        train_func: Callable[[Environment, BaseAgent, int], Iterable[EpisodeLog]],
        env: Environment,
        agent: BaseAgent,
        episodes: int = 15000
) -> list[float]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    print(f'Experiment name: {experiment_name}')
    print(f'Run name: {run_name}')

    training_rewards = []

    with mlflow.start_run(run_name=run_name):
        for name, value in agent.get_metrics().items():
            mlflow.log_param(name, value)

        logs = train_func(env, agent, episodes)

        for index, log in enumerate(logs):
            mlflow.log_metrics({
                "reward": log.summary_reward,
                "final_charge": log.final_charge,
                "episode_length": log.episode_length,
                "collision": log.collision_number,
            }, step=index)

            training_rewards.append(log.summary_reward)

    return training_rewards


def compute_metrics(
        experiment_name: str,
        run_name: str,
        test_func: Callable[[Environment, QLearningAgentBase, int, float], TestMetric],  # todo refactor?
        env: Environment,
        agent: QLearningAgentBase,
        episodes: int = 300,
        epsilon: float = 0.005
) -> None:
    mlflow.set_tracking_uri('http://localhost:5000')
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


def visualize_data(
    experiment_name: str,
    run_name: str,
    test_func: Callable[[Any, Any, int, float], tuple[Any, dict]],
    env: Any,
    agent: Any,
    episodes: int = 300,
    epsilon: float = 0.005,
    training_rewards: list[float] = None,
    smoothing_window: int = 150
) -> None:
    """
    Метод запускает тестирование агента, собирает сырые данные,
    генерирует графики в Matplotlib и сохраняет их в формате SVG в MLflow.
    """
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    print(f'Visualizing experiment: {experiment_name}')
    print(f'Run name: {run_name}')

    with mlflow.start_run(run_name=run_name):
        # Вызываем test_func, которая возвращает кортеж: (скалярные_метрики, словарь_сырых_данных)
        _, raw_data = test_func(env, agent, episodes, epsilon)

        # --- 1. Отрисовка и сохранение CDF ---
        fig_cdf = create_success_time_cdf_fig(raw_data.get("success_lengths", []))
        mlflow.log_figure(fig_cdf, "visualizations/success_time_cdf.svg")  # Заменено на .svg
        plt.close(fig_cdf)

        # --- 2. Отрисовка и сохранение Heatmap ---
        fig_heatmap = create_final_charge_heatmap_fig(
            final_x=raw_data.get("final_x", []),
            final_y=raw_data.get("final_y", []),
            final_charges=raw_data.get("final_charges", [])
        )
        mlflow.log_figure(fig_heatmap, "visualizations/final_charge_heatmap.svg")  # Заменено на .svg
        plt.close(fig_heatmap)

        # --- 3. Отрисовка и сохранение 4-х траекторий ---
        fig_traj = create_trajectories_fig(
            trajectories=raw_data.get("trajectories", []),
            agent_name=run_name
        )
        mlflow.log_figure(fig_traj, "visualizations/trajectories.svg")  # Заменено на .svg
        plt.close(fig_traj)

        # --- 4. Отрисовка и сохранение Кривой Обучения ---
        if training_rewards is not None:
            fig_learning = create_learning_curve_fig(training_rewards, window=smoothing_window)
            mlflow.log_figure(fig_learning, "visualizations/learning_curve.svg")  # Заменено на .svg
            plt.close(fig_learning)
            print("График кривой обучения успешно сохранен.")

        print(f"Все графики успешно сохранены в артефакты MLflow для '{run_name}'")


def evaluate_and_visualize(
    experiment_name: str,
    run_name: str,
    test_func: Callable[[Environment, QLearningAgentBase, int, float], tuple[TestMetric, dict]],
    env: Environment,
    agent: QLearningAgentBase,
    episodes: int = 300,
    epsilon: float = 0.005,
    training_rewards: list[float] = None,  # <--- Передаем сюда награды из log_data
    smoothing_window: int = 150            # <--- Окно сглаживания
) -> None:
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    print(f'Evaluating and visualizing: {run_name}')

    with mlflow.start_run(run_name=run_name):
        for name, value in agent.get_metrics().items():
            mlflow.log_param(name, value)

        metric, raw_data = test_func(env, agent, episodes, epsilon)

        # 1. Логируем числовые метрики
        mlflow.log_metrics({
            "success_percentage": metric.success_percentage,
            "average_success_episode_length": metric.average_success_episode_length,
            "average_charge_cost": metric.average_charge_cost,
            "average_episode_collision_percentage": metric.average_episode_collision_percentage,
        })

        # 2. CDF
        fig_cdf = create_success_time_cdf_fig(raw_data.get("success_lengths", []))
        mlflow.log_figure(fig_cdf, "visualizations/success_time_cdf.svg")
        plt.close(fig_cdf)

        # 3. Heatmap
        fig_heatmap = create_final_charge_heatmap_fig(
            final_x=raw_data.get("final_x", []),
            final_y=raw_data.get("final_y", []),
            final_charges=raw_data.get("final_charges", [])
        )
        mlflow.log_figure(fig_heatmap, "visualizations/final_charge_heatmap.svg")
        plt.close(fig_heatmap)

        # 4. Траектории
        fig_traj = create_trajectories_fig(
            trajectories=raw_data.get("trajectories", []),
            agent_name=run_name
        )
        mlflow.log_figure(fig_traj, "visualizations/trajectories.svg")
        plt.close(fig_traj)

        # === ДОБАВЛЕННЫЙ БЛОК: Кривая обучения ===
        if training_rewards is not None:
            fig_learning = create_learning_curve_fig(training_rewards, window=smoothing_window)
            mlflow.log_figure(fig_learning, "visualizations/learning_curve.svg")
            plt.close(fig_learning)
            print("График кривой обучения успешно сохранен в артефакты.")

        print(f"Метрики и все SVG-графики успешно сохранены в один Run для '{run_name}'")

def compare_agents_pipeline(
        experiment_name: str,
        env: Any,
        agent_a: Any,
        agent_b: Any,
        agent_a_name: str,
        agent_b_name: str,
        test_func: Any,
        episodes: int = 300,
        epsilon: float = 0.005
):
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    # Собираем данные тестов
    _, raw_data_a = test_func(env, agent_a, episodes, epsilon)
    _, raw_data_b = test_func(env, agent_b, episodes, epsilon)

    metric_to_compare = "battery_costs"
    samples_a = raw_data_a[metric_to_compare]
    samples_b = raw_data_b[metric_to_compare]

    # Считаем статистику чистым Python
    stats_results = perform_pure_python_ttest(
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        data_a=samples_a,
        data_b=samples_b,
        metric_name=metric_to_compare
    )

    # Создаем Run сравнения в MLflow
    run_name = f"Stat_Test_{agent_a_name}_vs_{agent_b_name}"
    with mlflow.start_run(run_name=run_name):
        # Логируем числовые параметры критерия
        mlflow.log_metrics({
            "t_statistic": stats_results["t_statistic"],
            "p_value": stats_results["p_value"],
            "significant_difference": stats_results["is_significant"]
        })

        # Сохраняем текстовый файл отчета как артефакт
        report_filename = "pure_python_statistical_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(stats_results["text_report"])

        mlflow.log_artifact(report_filename, artifact_path="statistics")
        print(f"Результаты сравнения успешно сохранены в MLflow Run: {run_name}")
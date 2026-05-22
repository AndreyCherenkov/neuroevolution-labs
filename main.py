import mlflow

from lab1.agents.prospect_agent import ProspectAgent
from lab1.agents.rational_q_learning import RationalQLearningAgent
from lab1.agents.risk_sensitive_agent import RiskSensitiveAgent
from lab1.env.agent_spawn import AgentSpawn
from lab1.env.environment import Environment
from lab1.env.obstacles import Obstacle
from lab1.env.recharge_station import RechargeStation
from lab1.env.wind import Wind
from lab1.tests.test import test_agents
from lab1.training.trainer import train
from lab1.visualisation.metrics import log_data, evaluate_and_visualize
from lab1.visualisation.t_test import perform_pure_python_ttest

if __name__ == '__main__':
    # 1. Инициализация общей для всех экспериментов среды
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

    # 2. Конфигурация пула исследуемых агентов
    agents_to_run = [
        (RationalQLearningAgent(9408, 12), "RationalQLearningAgent"),
        (ProspectAgent(9408, 12), "ProspectAgent2"),
        (RiskSensitiveAgent(9408, 12, eta=0.5), "RiskSensitiveAgent_0.5"),
        (RiskSensitiveAgent(9408, 12, eta=1), "RiskSensitiveAgent_1"),
        (RiskSensitiveAgent(9408, 12, eta=2), "RiskSensitiveAgent_2"),
        (RiskSensitiveAgent(9408, 12, eta=0.01), "RiskSensitiveAgent_0.01"),
        (RiskSensitiveAgent(9408, 12, eta=0.005), "RiskSensitiveAgent_0.0005")
    ]

    episodes_train = 45000
    episodes_test = 300
    test_epsilon = 0.005
    experiment_name = "DroneNavigation"

    # Базовый агент для попарного сравнения (Контрольная группа)
    control_agent_name = "RationalQLearningAgent"

    # Структура для сохранения сырых выборок (фиксированный размер N=300)
    all_agents_raw_metrics = {}

    print("=== Старт комплексного пайплайна: Обучение и Тестирование ===")

    # === ЭТАП 1: Обучение, валидация и логирование индивидуальных результатов ===
    for idx, (agent, base_name) in enumerate(agents_to_run, 1):
        print(f"\n[{idx}/{len(agents_to_run)}] Обработка архитектуры: {base_name}")
        print("-" * 50)

        # Шаг 1.1: Фаза обучения агента
        train_run_name = f"{base_name}_{episodes_train}"
        training_rewards = log_data(
            experiment_name=experiment_name,
            run_name=train_run_name,
            train_func=train,
            env=en,
            agent=agent,
            episodes=episodes_train
        )
        print(f"-> Обучение завершено. Собрано {len(training_rewards)} наград.")

        # Шаг 1.2: Фаза инференса и тестирования (сбор полных выборок)
        metric, raw_data = test_agents(
            env=en,
            agent=agent,
            episodes=episodes_test,
            epsilon=test_epsilon
        )

        # Шаг 1.3: Построение графиков и сохранение базовых метрик в MLflow
        test_run_name = f"{base_name}_test_{episodes_train}"
        evaluate_and_visualize(
            experiment_name=experiment_name,
            run_name=test_run_name,
            test_func=lambda env, agent, eps, eps_val: (metric, raw_data),  # Инжектируем уже готовые данные
            env=en,
            agent=agent,
            episodes=episodes_test,
            epsilon=test_epsilon,
            training_rewards=training_rewards
        )

        # Шаг 1.4: Архивация сырых распределений в оперативную память для t-теста
        all_agents_raw_metrics[base_name] = {
            "all_episode_lengths": raw_data.get("all_episode_lengths", []),
            "final_charges": raw_data.get("final_charges", []),
            "success_binary": raw_data.get("success_binary", []),
            "collision_binary": raw_data.get("collision_binary", [])
        }
        print(f"-> Тестирование и индивидуальное логирование для {base_name} успешно завершены.")

    # === ЭТАП 2: Попарный статистический анализ (Welch t-test) ===
    print("\n" + "=" * 60)
    print("Запуск многофакторного статистического анализа...")
    print("=" * 60)

    # Набор метрик для проверки статистических гипотез
    metrics_to_test = [
        "all_episode_lengths",  # Время/длина полёта (все эпизоды)
        "final_charges",  # Остаточный заряд батареи (все эпизоды)
        "success_binary",  # Вероятность успешного завершения задачи
        "collision_binary"  # Вероятность возникновения коллизии
    ]

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    # Генерируем выделенный агрегирующий Run для хранения стат-анализа всей серии тестов
    with mlflow.start_run(run_name=f"Statistical_Analysis_Welch_t_test_{episodes_train}"):

        for agent_name in all_agents_raw_metrics.keys():
            # Базовый агент не сравнивается сам с собой
            if agent_name == control_agent_name:
                continue

            print(f"\nСтатистическое сравнение: {control_agent_name} vs {agent_name}")
            print("-" * 50)

            for metric_key in metrics_to_test:
                data_control = all_agents_raw_metrics[control_agent_name][metric_key]
                data_treatment = all_agents_raw_metrics[agent_name][metric_key]

                # Защита от пустых выборок/исключений
                if len(data_control) < 2 or len(data_treatment) < 2:
                    print(f"[Пропуск] Метрика '{metric_key}': недостаточно данных для вычисления дисперсии.")
                    continue

                # Расчёт t-критерия Уэлча без внешних математических зависимостей
                test_results = perform_pure_python_ttest(
                    agent_a_name=control_agent_name,
                    agent_b_name=agent_name,
                    data_a=data_control,
                    data_b=data_treatment,
                    metric_name=metric_key,
                    alpha=0.05
                )

                # Логирование расчетных параметров теста в MLflow с уникальными префиксами
                run_prefix = f"{agent_name}_vs_Rational_{metric_key}"
                mlflow.log_metrics({
                    f"{run_prefix}_t_stat": test_results["t_statistic"],
                    f"{run_prefix}_p_value": test_results["p_value"],
                    f"{run_prefix}_significant": test_results["is_significant"]
                })

                # Сохранение физического текстового отчёта на диск и его отправка в артефакты
                report_filename = f"t_test_{agent_name}_vs_Rational_{metric_key}.txt"
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(test_results["text_report"])

                mlflow.log_artifact(report_filename, artifact_path="statistical_reports")

    print("\n" + "=" * 60)
    print("Пайплайны расчётов полностью завершены!")
    print("Все метрики, графики SVG и стат-отчёты успешно экспортированы в MLflow.")
    print("=" * 60)
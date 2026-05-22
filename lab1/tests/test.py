from lab1.agents.q_agent_base import QLearningAgentBase
from lab1.env.environment import Environment
from lab1.telemetry.test_metric import TestMetric


def test_agents(
        env: Environment,
        agent: QLearningAgentBase,
        episodes: int = 300,
        epsilon: float = 0.005
) -> tuple[TestMetric, dict]:
    old_epsilon = agent.epsilon
    agent.epsilon = epsilon

    successes = 0
    success_episode_length = 0
    total_energy_consumed = 0
    episodes_with_collisions = 0

    # --- Списки для сбора сырых данных для графиков ---
    success_lengths = []  # Длины только успешных эпизодов (для CDF)
    final_x = []  # Координаты X на момент завершения (для Heatmap)
    final_y = []  # Координаты Y на момент завершения (для Heatmap)
    final_charges = []  # Финальный заряд батареи (для Heatmap)
    trajectories = []  # Список траекторий для первых 4-х эпизодов

    # --- Дополнительные списки выборки для t-test ---
    all_episode_lengths = []  # Шаги абсолютно всех эпизодов
    success_binary = []  # 1.0 если успех, 0.0 если провал
    collision_binary = []  # 1.0 если было столкновение, 0.0 если нет

    for episode in range(episodes):
        state = env.reset()
        initial_battery = state.battery
        done = False

        episode_has_collision = False
        episode_success = False

        current_trajectory = []
        if episode < 4:
            current_trajectory.append((state.delta_x, state.delta_y))

        while not done:
            state_idx = state.discretize().to_index()
            action_ids = agent.act(state_idx)

            next_state, reward, done, is_success, is_collided = env.step(action_ids)

            if is_collided:
                episode_has_collision = True

            if is_success:
                episode_success = True

            state = next_state

            if episode < 4:
                current_trajectory.append((state.delta_x, state.delta_y))

        # Наполнение выборок для статистического анализа
        all_episode_lengths.append(float(env.time))
        success_binary.append(1.0 if episode_success else 0.0)
        collision_binary.append(1.0 if episode_has_collision else 0.0)

        # Стандартные агрегации
        if episode_success:
            successes += 1
            success_episode_length += env.time
            success_lengths.append(env.time)

        if episode_has_collision:
            episodes_with_collisions += 1

        total_energy_consumed += (initial_battery - state.battery)

        final_x.append(state.delta_x)
        final_y.append(state.delta_y)
        final_charges.append(state.battery)

        if episode < 4:
            trajectories.append(current_trajectory)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    # Расчет агрегированных метрик
    success_percentage = (successes / episodes) * 100

    average_success_episode_length = (
        success_episode_length / successes if successes > 0 else 0.0
    )

    average_charge_cost = total_energy_consumed / episodes

    average_episode_collision_percentage = (episodes_with_collisions / episodes) * 100

    metric = TestMetric(
        success_percentage=success_percentage,
        average_success_episode_length=average_success_episode_length,
        average_charge_cost=average_charge_cost,
        average_episode_collision_percentage=average_episode_collision_percentage
    )

    # Формируем итоговый словарь сырых данных
    raw_data = {
        # Данные для графиков (совместимость со старыми методами)
        "success_lengths": success_lengths,
        "final_x": final_x,
        "final_y": final_y,
        "final_charges": final_charges,
        "trajectories": trajectories,

        # Полноценные выборки фиксированной длины (N = 300) для t-тестов
        "all_episode_lengths": all_episode_lengths,
        "success_binary": success_binary,
        "collision_binary": collision_binary
    }

    return metric, raw_data

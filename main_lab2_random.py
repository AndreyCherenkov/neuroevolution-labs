import random
import numpy as np
import gymnasium as gym
import time

# ── ИМПОРТ MLFLOW ──────────────────────────────────────────────────────────
import mlflow

# ---------------------------------------------------------------------------
# КОНСТАНТЫ ОБУЧЕНИЯ (Полностью совпадают с основным алгоритмом)
# ---------------------------------------------------------------------------
POPULATION_SIZE     = 50
MAX_GENERATIONS     = 200
TARGET_BEST_FITNESS = 200.0
TARGET_AVG_FITNESS  = 0.0
STABILITY_PATIENCE  = 5
VISUALIZE_EVERY     = 25

RUN_NAME = "BASELINE Случайный агент (размер популяции 50)"


# ---------------------------------------------------------------------------
# ОЦЕНКА ОДНОЙ СЛУЧАЙНОЙ ОСОБИ
# ---------------------------------------------------------------------------
def evaluate_random_individual(env, episode_seed: int) -> float:
    obs, _ = env.reset(seed=episode_seed)
    total_reward = 0.0
    done = False

    while not done:
        # Выбираем абсолютно случайное действие (0, 1, 2, 3)
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, _ = env.step(action)

        # Кастомный reward shaping из оригинального скрипта main_lab2.py
        left_on_ground  = (obs[6] == 1.0)
        right_on_ground = (obs[7] == 1.0)
        both_on_ground  = left_on_ground and right_on_ground

        if both_on_ground and action == 0:
            reward += 0.5
        elif (left_on_ground or right_on_ground) and action != 0:
            reward -= 0.3

        total_reward += reward
        done = terminated or truncated

    return float(total_reward)


# ---------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ПОЛЁТА
# ---------------------------------------------------------------------------
def render_random_agent_episode(generation_label, seed=42):
    gen_str = str(generation_label)
    print(f"\n>>> Визуализация полёта случайного агента ({gen_str})...")

    render_env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = render_env.reset(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        action = int(render_env.action_space.sample())
        obs, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        time.sleep(0.01)
        done = terminated or truncated

    print(f">>> Полёт завершён. Reward: {total_reward:.2f}")
    render_env.close()


# ---------------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ БЕЙЗЛАЙНА
# ---------------------------------------------------------------------------
def run_baseline():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make("LunarLander-v3")
    env.action_space.seed(seed)

    stability_counter = 0

    # Константные параметры топологии для случайного агента (сеть не меняется)
    # 13 узлов (8 входов + 1 bias + 4 выхода) и 36 связей (9 * 4) как на старте NEAT
    nodes_cnt = 13
    conns_cnt = 36
    num_species = 1

    # ── НАСТРОЙКА MLFLOW ЭКСПЕРИМЕНТА ───────────────────────────────────────
    mlflow.set_experiment("NEAT_LunarLander_v3")

    print("=== Старт симуляции БЕЙЗЛАЙНА (Случайный Агент) ===")
    print(f"Цель: best_fitness >= {TARGET_BEST_FITNESS:.0f} И avg_fitness >= {TARGET_AVG_FITNESS:.0f} "
          f"на протяжении {STABILITY_PATIENCE} поколений подряд")
    print("-" * 80)

    with mlflow.start_run(run_name=RUN_NAME):

        # 1. Логирование параметров (аналогично NEAT для удобства сравнения)
        mlflow.log_params({
            "NUM_EVAL_EPISODES": 1,
            "TARGET_BEST_FITNESS": TARGET_BEST_FITNESS,
            "TARGET_AVG_FITNESS": TARGET_AVG_FITNESS,
            "STABILITY_PATIENCE": STABILITY_PATIENCE,
            "SEED": seed,
            "population_size": POPULATION_SIZE,
            "algorithm": "Pure Random Agent"
        })

        for generation in range(MAX_GENERATIONS):
            generation_fitnesses = []

            # Симулируем "популяцию" независимых случайных проходов
            for individual_idx in range(POPULATION_SIZE):
                # Генерируем уникальный сид для каждого полета внутри поколения
                episode_seed = seed + generation * 1000 + individual_idx
                fitness = evaluate_random_individual(env, episode_seed)
                generation_fitnesses.append(fitness)

            best_fitness = max(generation_fitnesses)
            avg_fitness  = float(np.mean(generation_fitnesses))

            # Вывод лога в консоль в оригинальном формате NEAT
            print(
                f"Gen {generation:3d} | "
                f"Best: {best_fitness:7.2f} | Avg: {avg_fitness:7.2f} | "
                f"Species: {num_species:2d} | "
                f"Topol (N/C): {nodes_cnt:2d}/{conns_cnt:3d} | "
                f"Stable: {stability_counter}/{STABILITY_PATIENCE}"
            )

            # 2. Логирование метрик поколения в MLflow (названия ключей совпадают 1-в-1)
            mlflow.log_metric("best_fitness", best_fitness, step=generation)
            mlflow.log_metric("avg_fitness", avg_fitness, step=generation)
            mlflow.log_metric("num_species", num_species, step=generation)
            mlflow.log_metric("nodes_count", nodes_cnt, step=generation)
            mlflow.log_metric("connections_count", conns_cnt, step=generation)
            mlflow.log_metric("stability_counter", stability_counter, step=generation)

            # Визуализация полета каждые N поколений
            if generation > 0 and generation % VISUALIZE_EVERY == 0:
                render_random_agent_episode(f"Поколение {generation}", seed=seed)
                print("-" * 80)

            # Проверка гипотетического условия успеха
            if best_fitness >= TARGET_BEST_FITNESS and avg_fitness >= TARGET_AVG_FITNESS:
                stability_counter += 1
                if stability_counter >= STABILITY_PATIENCE:
                    print("\n[УСПЕХ] Случайный агент невероятным образом достиг цели!")
                    break
            else:
                stability_counter = 0

        env.close()
        mlflow.set_tag("status", "completed")
        mlflow.log_metric("final_best_fitness", best_fitness)

    print("\nБейзлайн скрипт завершён.")


if __name__ == "__main__":
    run_baseline()
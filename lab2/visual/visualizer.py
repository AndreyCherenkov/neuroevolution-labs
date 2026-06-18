import pickle
import sys
import time
from typing import List

import gymnasium as gym
import numpy as np

from lab2.neat.neural_network import NeuralNetwork


def run_saved_agent(
    model_path: str = "lab2/best_neat_genome.pkl",
    num_episodes: int = 5,
    seed: int = 42
) -> None:
    """
    Загружает сохраненный геном NEAT и визуализирует работу агента в LunarLander-v3.

    Args:
        model_path (str): Путь к сериализованному файлу (.pkl) с лучшим геномом.
        num_episodes (int): Количество демонстрационных эпизодов.
        seed (int): Базовое начальное значение генератора случайных чисел для среды.
    """
    print(f"=== Загрузка модели из файла: {model_path} ===")
    try:
        with open(model_path, "rb") as f:
            genome = pickle.load(f)
        print(f"[УСПЕХ] Модель загружена. Фитнес модели при обучении: {genome.fitness:.2f}")
    except FileNotFoundError:
        print(f"[ОШИБКА] Файл {model_path} не найден. Сначала запустите обучение.")
        return
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить модель: {e}")
        return

    print(f"\nИнициализация окружения LunarLander-v3 для {num_episodes} эпизодов...")
    env = gym.make("LunarLander-v3", render_mode="human")

    # Инициализация фенотипа (нейросети) из генотипа
    nn = NeuralNetwork(genome)

    for episode in range(num_episodes):
        current_seed = seed + episode
        obs, _ = env.reset(seed=current_seed)
        total_reward = 0.0
        done = False
        steps = 0

        print(f"\n► Старт эпизода {episode + 1}/{num_episodes} (Seed: {current_seed})")

        while not done:
            # Активация сети и выбор действия с максимальным сигналом
            outputs: List[float] = nn.activate(obs.tolist())
            action = int(np.argmax(outputs))

            # Шаг в симуляции окружения
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Стабилизация FPS отображения
            time.sleep(0.01)
            done = terminated or truncated

        print(f"Эпизод {episode + 1} завершен за {steps} шагов. Набранный Reward: {total_reward:.2f}")

    print("\n=== Демонстрация успешно завершена ===")
    env.close()


if __name__ == "__main__":
    # Глобальные параметры по умолчанию
    DEFAULT_MODEL = "C:/Users/Andrey/PycharmProjects/NeuroevolutionaryComputations/lab2/best_neat_genome.pkl"
    DEFAULT_EPISODES = 10

    episodes_count = DEFAULT_EPISODES
    target_model_path = DEFAULT_MODEL

    if len(sys.argv) > 1:
        try:
            episodes_count = int(sys.argv[1])
        except ValueError:
            print(
                f"[Внимание] Неверный формат количества эпизодов. "
                f"Используем значение по умолчанию: {DEFAULT_EPISODES}"
            )

    if len(sys.argv) > 2:
        target_model_path = sys.argv[2]

    run_saved_agent(model_path=target_model_path, num_episodes=episodes_count)
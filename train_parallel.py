"""
train_parallel.py
=================
Параллельная версия обучения NEAT с поддержкой параллельной оценки фитнеса.

Ключевые изменения по сравнению с оригиналом:
  - evaluate_population(genomes, env, ..., parallel=True/False)
    При parallel=True каждый геном оценивается в отдельном процессе через
    ProcessPoolExecutor. Каждый воркер создаёт собственный gym.Env, поэтому
    передавать env между процессами не нужно.
  - run_training(parallel=True) — флаг пробрасывается до evaluate_population
    и выводится в заголовке запуска.
"""

import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.lines import Line2D

from lab2.neat.genome import Genome
from lab2.neat.innovation_tracker import InnovationTracker
from lab2.neat.neat_config import NEATConfig
from lab2.neat.neural_network import NeuralNetwork
from lab2.neat.node_id_tracker import NodeIdTracker
from lab2.neat.node_type import NodeType
from lab2.neat.population import Population
from lab2.neat.population_factory import PopulationFactory

# ---------------------------------------------------------------------------
# КОНСТАНТЫ ОБУЧЕНИЯ
# ---------------------------------------------------------------------------
NUM_EVAL_EPISODES: int = 30
TARGET_BEST_FITNESS: float = 200.0
TARGET_AVG_FITNESS: float = -30.0
STABILITY_PATIENCE: int = 5
VISUALIZE_EVERY: int = 25

RUN_NAME: str = "NEAT parallel eval (популяция 300)"
MODEL_SAVE_PATH: str = "lab2/best_neat_genome_parallel.pkl"

# Метки узлов LunarLander-v3
INPUT_LABELS: List[str] = [
    "x pos", "y pos",
    "x vel", "y vel",
    "angle", "ang vel",
    "leg L", "leg R",
]
OUTPUT_LABELS: List[str] = ["noop", "left eng", "main eng", "right eng"]


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОЦЕНКИ В ОТДЕЛЬНОМ ПРОЦЕССЕ
# ---------------------------------------------------------------------------
def _evaluate_genome_worker(args: tuple) -> tuple[int, float]:
    """
    Воркер для параллельной оценки одного генома.

    Функция запускается в отдельном процессе, поэтому создаёт собственное
    gymnasium-окружение и не зависит от env основного процесса.

    Args:
        args: Кортеж (genome_index, genome, base_seed, num_episodes).
              genome_index нужен для сборки результатов в правильном порядке.

    Returns:
        Кортеж (genome_index, fitness).
    """
    genome_index, genome, base_seed, num_episodes = args

    # Локальное окружение внутри воркера
    env = gym.make("LunarLander-v3")
    env.action_space.seed(base_seed)

    nn = NeuralNetwork(genome)
    episode_rewards: List[float] = []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + episode_idx)
        total_reward = 0.0
        done = False

        while not done:
            outputs = nn.activate(obs.tolist())
            action = int(np.argmax(outputs))
            obs, reward, terminated, truncated, _ = env.step(action)

            left_on_ground = obs[6] == 1.0
            right_on_ground = obs[7] == 1.0
            both_on_ground = left_on_ground and right_on_ground

            if both_on_ground and action == 0:
                reward += 0.5
            elif (left_on_ground or right_on_ground) and action != 0:
                reward -= 0.3

            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    env.close()
    return genome_index, float(np.mean(episode_rewards))


# ---------------------------------------------------------------------------
# ОЦЕНКА ОДНОГО ГЕНОМА (однопоточный режим)
# ---------------------------------------------------------------------------
def evaluate_genome(genome: Genome, env: gym.Env, base_seed: int,
                    num_episodes: int = NUM_EVAL_EPISODES) -> float:
    """
    Однопоточная оценка генома: средний reward за num_episodes эпизодов.

    Args:
        genome: Оцениваемый геном.
        env: Уже созданное gymnasium-окружение.
        base_seed: Базовый seed для воспроизводимости.
        num_episodes: Количество эпизодов.

    Returns:
        Средний reward (fitness).
    """
    nn = NeuralNetwork(genome)
    episode_rewards: List[float] = []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + episode_idx)
        total_reward = 0.0
        done = False

        while not done:
            outputs = nn.activate(obs.tolist())
            action = int(np.argmax(outputs))
            obs, reward, terminated, truncated, _ = env.step(action)

            left_on_ground = obs[6] == 1.0
            right_on_ground = obs[7] == 1.0
            both_on_ground = left_on_ground and right_on_ground

            if both_on_ground and action == 0:
                reward += 0.5
            elif (left_on_ground or right_on_ground) and action != 0:
                reward -= 0.3

            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards))


# ---------------------------------------------------------------------------
# ОЦЕНКА ВСЕЙ ПОПУЛЯЦИИ — ЕДИНАЯ ТОЧКА С ФЛАГОМ ПАРАЛЛЕЛЬНОСТИ
# ---------------------------------------------------------------------------
def evaluate_population(
        genomes: List[Genome],
        env: gym.Env,
        base_seed: int,
        num_episodes: int = NUM_EVAL_EPISODES,
        parallel: bool = True,
        max_workers: Optional[int] = None,
) -> List[float]:
    """
    Оценивает приспособленность всех геномов популяции.

    Args:
        genomes: Список геномов текущего поколения.
        env: Gymnasium-окружение (используется только при parallel=False).
        base_seed: Базовый seed для воспроизводимости.
        num_episodes: Количество эпизодов на геном.
        parallel: Если True — параллельная оценка через ProcessPoolExecutor;
                  если False — последовательная однопоточная оценка.
        max_workers: Максимальное число параллельных воркеров.
                     По умолчанию — os.cpu_count().

    Returns:
        Список значений fitness в том же порядке, что и входной список геномов.
    """
    fitnesses: List[float] = [0.0] * len(genomes)

    if parallel:
        # Параллельный режим: каждый воркер создаёт собственный env
        workers = max_workers or os.cpu_count()
        args_list = [
            (idx, genome, base_seed, num_episodes)
            for idx, genome in enumerate(genomes)
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_evaluate_genome_worker, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                genome_index, fitness = future.result()
                fitnesses[genome_index] = fitness
    else:
        # Однопоточный режим: используем переданный env
        for idx, genome in enumerate(genomes):
            fitnesses[idx] = evaluate_genome(genome, env, base_seed, num_episodes)

    return fitnesses


# ---------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ТОПОЛОГИИ СЕТИ
# ---------------------------------------------------------------------------
def _compute_node_positions(genome, nn=None):
    """Вычисляет координаты (x, y) для каждого узла графа нейронной сети."""
    input_nodes = sorted([n for n in genome.nodes.values() if n.node_type == NodeType.INPUT], key=lambda n: n.node_id)
    bias_nodes = [n for n in genome.nodes.values() if n.node_type == NodeType.BIAS]
    output_nodes = sorted([n for n in genome.nodes.values() if n.node_type == NodeType.OUTPUT],
                          key=lambda n: n.node_id)
    hidden_nodes = sorted([n for n in genome.nodes.values() if n.node_type == NodeType.HIDDEN],
                          key=lambda n: n.node_id)

    pos = {}

    for i, node in enumerate(input_nodes + bias_nodes):
        y = 1.0 - i / max(len(input_nodes + bias_nodes) - 1, 1)
        pos[node.node_id] = (0.0, y)

    for i, node in enumerate(output_nodes):
        y = 1.0 - i / max(len(output_nodes) - 1, 1)
        pos[node.node_id] = (1.0, y)

    if hidden_nodes and nn is not None:
        topo = nn.topological_order
        hidden_ids_ordered = [nid for nid in topo if nid in {n.node_id for n in hidden_nodes}]
        n_h = len(hidden_ids_ordered)
        for rank, nid in enumerate(hidden_ids_ordered):
            x = 0.25 + 0.5 * rank / max(n_h - 1, 1)
            y = 0.5 + 0.35 * np.sin(rank * 1.3)
            pos[nid] = (x, y)
    elif hidden_nodes:
        for i, node in enumerate(hidden_nodes):
            pos[node.node_id] = (0.5, 0.9 - i * 0.2)

    return pos, input_nodes, bias_nodes, output_nodes, hidden_nodes


def visualize_network_structure(genome: Genome, title: str = "Топология сети",
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Отрисовывает ориентированный граф нейронной сети текущего генома."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 7))

    try:
        nn = NeuralNetwork(genome)
    except Exception:
        nn = None

    pos, _, _, _, _ = _compute_node_positions(genome, nn)

    enabled_conns = [c for c in genome.connections.values() if c.is_enabled]
    disabled_conns = [c for c in genome.connections.values() if not c.is_enabled]

    for conn in disabled_conns:
        if conn.in_node_id not in pos or conn.out_node_id not in pos:
            continue
        x0, y0 = pos[conn.in_node_id]
        x1, y1 = pos[conn.out_node_id]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#cccccc", lw=0.5, connectionstyle="arc3,rad=0.08"))

    if enabled_conns:
        weights = np.array([c.weight for c in enabled_conns])
        w_abs = np.abs(weights)
        w_norm = w_abs / max(w_abs.max(), 1e-9)

        for i, conn in enumerate(enabled_conns):
            if conn.in_node_id not in pos or conn.out_node_id not in pos:
                continue
            x0, y0 = pos[conn.in_node_id]
            x1, y1 = pos[conn.out_node_id]
            w = weights[i]
            lw = 0.6 + 2.4 * w_norm[i]
            color = "#d43a3a" if w > 0 else "#3a6fd4"
            alpha = 0.35 + 0.55 * w_norm[i]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                        connectionstyle="arc3,rad=0.08"))

    node_r = 0.028
    label_x_offset_in = -0.07
    label_x_offset_out = 0.07

    node_style = {
        NodeType.INPUT: ("#2ecc71", "#1a7a43", "left", INPUT_LABELS),
        NodeType.BIAS: ("#f39c12", "#a0620a", "left", ["bias"]),
        NodeType.OUTPUT: ("#e74c3c", "#8b1a1a", "right", OUTPUT_LABELS),
        NodeType.HIDDEN: ("#9b59b6", "#5b2c6f", None, []),
    }

    for node in genome.nodes.values():
        if node.node_id not in pos:
            continue
        x, y = pos[node.node_id]
        fc, ec, side, labels = node_style[node.node_type]
        circle = plt.Circle((x, y), node_r, color=fc, ec=ec, lw=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, str(node.node_id), ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=5)

        if side in ("left", "right"):
            same_type = sorted([n for n in genome.nodes.values() if n.node_type == node.node_type],
                               key=lambda n: n.node_id)
            idx = same_type.index(node)
            if side == "left":
                label_text = labels[idx] if idx < len(labels) else f"in{node.node_id}"
                ax.text(x + label_x_offset_in, y, label_text, ha="right", va="center", fontsize=8, color="#333333")
            else:
                label_text = labels[idx] if idx < len(labels) else f"out{node.node_id}"
                ax.text(x + label_x_offset_out, y, label_text, ha="left", va="center", fontsize=8, color="#333333")

    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="#1a7a43", label="Входной"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="#a0620a", label="Bias"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#8b1a1a", label="Выходной"),
        mpatches.Patch(facecolor="#9b59b6", edgecolor="#5b2c6f", label="Скрытый"),
        Line2D([0], [0], color="#d43a3a", lw=2, label="Вес > 0"),
        Line2D([0], [0], color="#3a6fd4", lw=2, label="Вес < 0"),
        Line2D([0], [0], color="#cccccc", lw=1, linestyle="--", label="Отключена"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=8, framealpha=0.7,
              bbox_to_anchor=(0.5, -0.04))

    enabled_count = len(enabled_conns)
    ax.set_title(
        f"{title}\nФитнес: {genome.fitness:.2f} | Узлов: {len(genome.nodes)} | Связей (вкл): {enabled_count}",
        fontsize=11, pad=10
    )
    ax.set_xlim(-0.22, 1.18)
    ax.set_ylim(-0.15, 1.12)
    ax.axis("off")
    ax.set_aspect("equal")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def visualize_comparison(genome_initial: Genome, genome_best: Genome) -> None:
    """Отрисовывает начальную и лучшую топологии сетей бок о бок для сравнения."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Эволюция топологии нейронной сети", fontsize=14, fontweight="bold", y=1.01)
    visualize_network_structure(genome_initial, title="Начальная топология (Gen 0)", ax=axes[0])
    visualize_network_structure(genome_best, title="Лучшая топология", ax=axes[1])
    plt.tight_layout()
    plt.show()


def render_best_agent_episode(genome: Genome, generation_label: str, seed: int = 42) -> None:
    """Запускает демонстрационный визуальный эпизод с фиксированным seed."""
    print(f"\n>>> Визуализация полёта агента ({generation_label})...")
    render_env = gym.make("LunarLander-v3", render_mode="human")
    nn = NeuralNetwork(genome)
    obs, _ = render_env.reset(seed=seed)
    render_env.action_space.seed(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        outputs = nn.activate(obs.tolist())
        action = int(np.argmax(outputs))
        obs, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        time.sleep(0.01)
        done = terminated or truncated

    print(f">>> Полёт завершён. Reward: {total_reward:.2f}")
    render_env.close()


# ---------------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ С ФЛАГОМ ПАРАЛЛЕЛЬНОСТИ
# ---------------------------------------------------------------------------
def run_training(parallel: bool = True, max_workers: Optional[int] = None) -> None:
    """
    Запускает эволюционный процесс NEAT.

    Args:
        parallel: True  — параллельная оценка фитнеса через ProcessPoolExecutor
                          (каждый воркер создаёт собственный gym.Env).
                  False — последовательная однопоточная оценка (оригинальное поведение).
        max_workers: Число параллельных воркеров при parallel=True.
                     По умолчанию — количество логических ядер CPU.
    """
    mode_label = (
        f"ПАРАЛЛЕЛЬНЫЙ (воркеров: {max_workers or os.cpu_count()})"
        if parallel else "ОДНОПОТОЧНЫЙ"
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    config = NEATConfig(
        population_size=300,
        compatibility_threshold=1.0,
        c1=1.0, c2=1.0, c3=0.3,
        max_stagnation=15,
        elite_fraction=0.1,
        tournament_size=3,
        prob_mutate_weight=0.8,
        prob_weight_replace=0.1,
        weight_mutation_power=0.5,
        weight_min=-3.0,
        weight_max=3.0,
        mutate_weight_min=-5.0,
        mutate_weight_max=5.0,
        prob_add_node=0.04,
        prob_add_conn=0.15,
        prob_enable_connection=0.02,
        prob_disable_connection=0.01,
    )

    num_inputs = 8
    num_outputs = 4

    innovation_tracker = InnovationTracker()
    start_node_id = num_inputs + 1 + num_outputs - 1
    node_tracker = NodeIdTracker(start_id=start_node_id)

    initial_genomes = PopulationFactory.create_initial_population(
        population_size=config.population_size,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        innovation_tracker=innovation_tracker,
    )

    population = Population(
        genomes=initial_genomes,
        config=config,
        innovation_tracker=innovation_tracker,
        node_tracker=node_tracker,
    )

    # env нужен только для однопоточного режима и визуализации
    env = gym.make("LunarLander-v3")
    env.action_space.seed(seed)

    max_generations = 200
    stability_counter = 0
    best_genome_ever: Optional[Genome] = None

    initial_genome_snapshot = initial_genomes[0].clone()

    mlflow.set_experiment("NEAT_LunarLander_v3")

    print("=== Старт нейроэволюционного обучения NEAT ===")
    print(f"Режим оценки фитнеса: {mode_label}")
    print(f"Цель: best_fitness >= {TARGET_BEST_FITNESS:.0f} И avg_fitness >= {TARGET_AVG_FITNESS:.0f} "
          f"на протяжении {STABILITY_PATIENCE} поколений подряд")
    print("-" * 80)

    with mlflow.start_run(run_name=f"{RUN_NAME} [{mode_label}]"):
        mlflow.log_params({
            "NUM_EVAL_EPISODES": NUM_EVAL_EPISODES,
            "TARGET_BEST_FITNESS": TARGET_BEST_FITNESS,
            "TARGET_AVG_FITNESS": TARGET_AVG_FITNESS,
            "STABILITY_PATIENCE": STABILITY_PATIENCE,
            "SEED": seed,
            "parallel_eval": parallel,
            "max_workers": max_workers or os.cpu_count(),
        })
        neat_params = {attr: getattr(config, attr) for attr in dir(config)
                       if not attr.startswith("__") and not callable(getattr(config, attr))}
        mlflow.log_params(neat_params)

        fig_init, ax_init = plt.subplots(figsize=(11, 7))
        visualize_network_structure(initial_genome_snapshot, title="Начальная топология (Gen 0)", ax=ax_init)
        mlflow.log_figure(fig_init, "topologies/initial_topology.png")
        plt.close(fig_init)

        for generation in range(max_generations):
            t_start = time.perf_counter()

            # ── КЛЮЧЕВОЕ МЕСТО: единый вызов с флагом parallel ──────────────
            fitnesses = evaluate_population(
                genomes=population.genomes,
                env=env,
                base_seed=seed,
                num_episodes=NUM_EVAL_EPISODES,
                parallel=parallel,
                max_workers=max_workers,
            )
            # ────────────────────────────────────────────────────────────────

            # Присваиваем fitness геномам
            for genome, fitness in zip(population.genomes, fitnesses):
                genome.fitness = fitness

            eval_time = time.perf_counter() - t_start

            best_fitness = max(fitnesses)
            avg_fitness = float(np.mean(fitnesses))
            best_genome_in_gen = max(population.genomes, key=lambda g: g.fitness)

            if best_genome_ever is None or best_genome_in_gen.fitness > best_genome_ever.fitness:
                best_genome_ever = best_genome_in_gen.clone()

            nodes_cnt = len(best_genome_in_gen.nodes)
            conns_cnt = len([c for c in best_genome_in_gen.connections.values() if c.is_enabled])

            print(
                f"Gen {generation:3d} | "
                f"Best: {best_fitness:7.2f} | Avg: {avg_fitness:7.2f} | "
                f"Species: {len(population.species):2d} | "
                f"Topol (N/C): {nodes_cnt:2d}/{conns_cnt:3d} | "
                f"Stable: {stability_counter}/{STABILITY_PATIENCE} | "
                f"Eval: {eval_time:.1f}s [{mode_label[:4]}]"
            )

            mlflow.log_metric("best_fitness", best_fitness, step=generation)
            mlflow.log_metric("avg_fitness", avg_fitness, step=generation)
            mlflow.log_metric("num_species", len(population.species), step=generation)
            mlflow.log_metric("nodes_count", nodes_cnt, step=generation)
            mlflow.log_metric("connections_count", conns_cnt, step=generation)
            mlflow.log_metric("stability_counter", stability_counter, step=generation)
            mlflow.log_metric("eval_time_seconds", eval_time, step=generation)

            if generation > 0 and generation % VISUALIZE_EVERY == 0:
                render_best_agent_episode(best_genome_in_gen, f"Поколение {generation}", seed=seed)
                print("-" * 80)

            if best_fitness >= TARGET_BEST_FITNESS and avg_fitness >= TARGET_AVG_FITNESS:
                stability_counter += 1
                if stability_counter >= STABILITY_PATIENCE:
                    print(
                        f"\n[УСПЕХ] Комплексная цель достигнута: "
                        f"best_fitness >= {TARGET_BEST_FITNESS:.1f} и avg_fitness >= {TARGET_AVG_FITNESS:.1f} "
                        f"удерживаются {STABILITY_PATIENCE} поколений подряд."
                    )
                    break
            else:
                stability_counter = 0

            population.evolve()

        env.close()

        if best_genome_ever:
            print("\n" + "=" * 80)
            print("ОБУЧЕНИЕ ЗАВЕРШЕНО. СОХРАНЕНИЕ МОДЕЛИ...")
            print("=" * 80)

            mlflow.set_tag("status", "completed")
            mlflow.log_metric("final_best_fitness", best_genome_ever.fitness)

            try:
                with open(MODEL_SAVE_PATH, "wb") as f:
                    pickle.dump(best_genome_ever, f)
                print(f"[УСПЕХ] Модель сохранена локально: {MODEL_SAVE_PATH}")
                mlflow.log_artifact(MODEL_SAVE_PATH)
                print(f"[MLFLOW] Артефакт добавлен в ран.")
            except Exception as e:
                print(f"[ОШИБКА] Не удалось сохранить модель: {e}")

            fig_best, ax_best = plt.subplots(figsize=(11, 7))
            visualize_network_structure(best_genome_ever, title="Лучшая топология сети за всё время", ax=ax_best)
            mlflow.log_figure(fig_best, "topologies/best_topology.png")
            plt.close(fig_best)

            fig_comp, axes_comp = plt.subplots(1, 2, figsize=(20, 8))
            fig_comp.suptitle("Эволюция топологии нейронной сети (Сравнение)", fontsize=14, fontweight="bold",
                               y=1.01)
            visualize_network_structure(initial_genome_snapshot, title="Начальная топология (Gen 0)",
                                        ax=axes_comp[0])
            visualize_network_structure(
                best_genome_ever, title=f"Лучшая топология (Фитнес: {best_genome_ever.fitness:.2f})",
                ax=axes_comp[1]
            )
            mlflow.log_figure(fig_comp, "topologies/comparison_topology.png")
            plt.close(fig_comp)

            visualize_comparison(initial_genome_snapshot, best_genome_ever)

    print("\nСкрипт завершён.")


# ---------------------------------------------------------------------------
# ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEAT LunarLander training")
    parser.add_argument(
        "--parallel",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help="Режим оценки фитнеса: True — параллельный, False — однопоточный (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Число воркеров при параллельном режиме (default: cpu_count)",
    )
    args = parser.parse_args()

    # ProcessPoolExecutor требует защиты точки входа на Windows/macOS
    # run_training(parallel=args.parallel, max_workers=args.workers)

    run_training(parallel=True, max_workers=8)
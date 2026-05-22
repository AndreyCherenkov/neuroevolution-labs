import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def create_learning_curve_fig(rewards: list, window: int = 150) -> plt.Figure:
    """Генерирует кривую обучения агента в объектно-ориентированном стиле Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5))

    smoothed_rewards = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    # Сырые награды (полупрозрачная линия)
    ax.plot(rewards, alpha=0.3, color='blue', label='Награда за эпизод')
    # Сглаженный тренд
    ax.plot(smoothed_rewards, color='red', linewidth=2, label=f'Скользящее среднее (окно {window})')

    ax.set_title('Кривая обучения агента')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Суммарная награда (Reward)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    return fig


def create_trajectories_fig(trajectories: list, agent_name: str, is_relative: bool = True) -> plt.Figure:
    """
    Генерирует карту среды с препятствиями, зонами спавна и траекториями движения дрона.

    :param trajectories: список эпизодов, где каждый эпизод — список позиций (координат).
    :param agent_name: имя агента для отображения в заголовке.
    :param is_relative: если True, то координаты трактуются как (delta_x, delta_y)
                        и переводятся в абсолютные относительно станции (21.5, 8.5).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Координаты зарядной станции из Варианта 1
    st_x, st_y = 21.5, 8.5

    # --- 1. Отрисовка препятствий (Вариант 1) ---
    obstacles = [
        patches.Rectangle((10.0, 2.0), 1.5, 3.5, color='gray', alpha=0.7, hatch='//'),
        patches.Rectangle((13.0, 7.0), 1.5, 3.0, color='gray', alpha=0.7, hatch='//'),
        patches.Rectangle((16.5, 3.5), 1.5, 3.0, color='gray', alpha=0.7, hatch='//'),
        patches.Rectangle((19.0, 9.0), 1.5, 2.5, color='gray', alpha=0.7, hatch='//'),
    ]
    for obs in obstacles:
        ax.add_patch(obs)

    # --- 2. Отрисовка зарядной станции ---
    station = patches.Circle((st_x, st_y), 0.9, color='green', alpha=0.4, label='Зарядная станция')
    ax.add_patch(station)
    ax.plot(st_x, st_y, 'g+', markersize=10)

    # --- 3. Отрисовка геометрии начальной позиции (Зона спавна дрона) ---
    # Точка математического ожидания старта \mu = (3.5, 6.0)
    ax.plot(3.5, 6.0, marker='o', color='black', markersize=6, label='Среднее старта')

    # Штрихпунктирные границы области усечения (клиппинга): x от 1 до 7 (ширина 6), y от 2 до 10 (высота 8)
    spawn_clip = patches.Rectangle(
        (1.0, 2.0), 6.0, 8.0,
        linewidth=1.2, edgecolor='dimgray', linestyle='-.', facecolor='none',
        label='Область усечения старта'
    )
    ax.add_patch(spawn_clip)

    # Пунктирная область 2\sigma вокруг \mu (радиус = 2 * 1.8 = 3.6)
    sigma_circle = patches.Circle(
        (3.5, 6.0), 3.6,
        linewidth=1.0, edgecolor='gray', linestyle='--', facecolor='none',
        label='Область старта'
    )
    ax.add_patch(sigma_circle)

    # --- 4. Отрисовка траекторий дрона ---
    colors = ['red', 'blue', 'orange', 'purple']
    for idx, traj in enumerate(trajectories[:4]):
        if is_relative:
            # Пересчет из относительных (delta_x, delta_y) в абсолютные (x, y)
            # Если у вас в среде delta = x_agent - x_station, поменяйте минус на плюс
            # (предполагаем классическое смещение: delta = x_station - x_agent)
            traj_x = [st_x - pos[0] for pos in traj]
            traj_y = [st_y - pos[1] for pos in traj]
        else:
            traj_x = [pos[0] for pos in traj]
            traj_y = [pos[1] for pos in traj]

        # Сама линия траектории
        ax.plot(traj_x, traj_y, color=colors[idx], linewidth=1.5, alpha=0.8, label=f'Траектория {idx + 1}')
        # Старт (зеленый круг)
        ax.plot(traj_x[0], traj_y[0], 'go', markersize=5)
        # Финиш (черный крестик)
        ax.plot(traj_x[-1], traj_y[-1], 'kx', markersize=6)

    # Настройки осей и сетки в соответствии с границами Варианта 1 (24х12)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.set_title(f'Примеры траекторий движения агента ({agent_name})')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Сдвигаем легенду вбок, чтобы она не перекрывала график из-за обилия новых объектов
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    return fig


def create_success_time_cdf_fig(success_lengths: list) -> plt.Figure:
    """Генерирует фигуру эмпирической функции распределения (CDF) времени успеха."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if not success_lengths:
        ax.text(
            0.5, 0.5,
            "Нет успешных эпизодов для расчета CDF",
            ha='center', va='center', fontsize=12, color='gray'
        )
        ax.set_title('CDF времени до успеха')
        return fig

    sorted_data = np.sort(success_lengths)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    ax.step(sorted_data, y_values, where='post', color='darkgreen', linewidth=2)

    ax.set_title('CDF времени (количества шагов) до успеха')
    ax.set_xlabel('Длина успешного эпизода (шаги)')
    ax.set_ylabel('F(t) — Вероятность завершения')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig


def create_final_charge_heatmap_fig(final_x: list, final_y: list, final_charges: list) -> plt.Figure:
    """Генерирует тепловую карту распределения финального заряда."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if not final_x or not final_y:
        ax.text(
            0.5, 0.5,
            "Нет данных для построения тепловой карты",
            ha='center', va='center', fontsize=12, color='gray'
        )
        return fig

    x_bins = 14
    y_bins = 14

    heatmap_sum = np.zeros((y_bins, x_bins))
    heatmap_count = np.zeros((y_bins, x_bins))

    x_step = 24 / x_bins
    y_step = 12 / y_bins

    for x, y, charge in zip(final_x, final_y, final_charges):
        # Ограничиваем индексы от 0 до bins-1, чтобы избежать ошибок
        # при вылете дрона за границы (например, x < 0 из-за ветра)
        x_idx = max(0, min(int(x // x_step), x_bins - 1))
        y_idx = max(0, min(int(y // y_step), y_bins - 1))

        heatmap_sum[y_idx, x_idx] += charge
        heatmap_count[y_idx, x_idx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = heatmap_sum / heatmap_count
        heatmap_data[heatmap_count == 0] = np.nan

    # Копируем палитру и задаем светло-серый цвет для ячеек без данных (NaN)
    cmap = plt.colormaps.get_cmap('RdYlGn').copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(
        heatmap_data,
        origin='lower',
        cmap=cmap,
        aspect='auto',
        extent=(0, 24, 0, 12)
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Средний финальный заряд')

    ax.set_title('Тепловая карта финального заряда батареи в точках завершения')
    ax.set_xlabel('Координата X')
    ax.set_ylabel('Координата Y')

    ax.set_xticks(np.arange(0, 25, 1))
    ax.set_yticks(np.arange(0, 13, 1))
    ax.grid(False)
    plt.tight_layout()

    return fig
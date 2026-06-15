from dataclasses import dataclass


@dataclass
class NEATConfig:
    """
    Конфигурация параметров алгоритма NEAT (NeuroEvolution of Augmenting Topologies).

    Attributes:
        population_size (int): Размер популяции особей.
        compatibility_threshold (float): Порог совместимости для разделения на виды (видообразование).
        c1 (float): Коэффициент избыточных (excess) генов при расчете генетического расстояния.
        c2 (float): Коэффициент разделенных (disjoint) генов при расчете генетического расстояния.
        c3 (float): Коэффициент разности весов при расчете генетического расстояния.
        max_stagnation (int): Максимальное число поколений без улучшений до стагнации вида.
        elite_fraction (float): Доля лучших особей вида, проходящих в следующее поколение.
        tournament_size (int): Количество особей, участвующих в турнирном отборе.
        prob_mutate_weight (float): Вероятность мутации весов соединений.
        prob_weight_replace (float): Вероятность полной замены веса вместо его пертурбации.
        prob_add_node (float): Вероятность структурной мутации добавления нового узла.
        prob_add_conn (float): Вероятность структурной мутации добавления новой связи.
        prob_enable_connection (float): Вероятность принудительного включения связи.
        prob_disable_connection (float): Вероятность принудительного отключения связи.
        weight_mutation_power (float): Мощность (СКО) гауссова шума при пертурбации весов.
        weight_min (float): Минимальный начальный вес для новых соединений.
        weight_max (float): Максимальный начальный вес для новых соединений.
        mutate_weight_min (float): Минимальная нижняя граница веса при мутации.
        mutate_weight_max (float): Максимальная верхняя граница веса при мутации.
    """

    population_size: int = 150
    compatibility_threshold: float = 0.8
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.5
    max_stagnation: int = 20
    elite_fraction: float = 0.1
    tournament_size: int = 3
    prob_mutate_weight: float = 0.8
    prob_weight_replace: float = 0.1
    prob_add_node: float = 0.10
    prob_add_conn: float = 0.20
    prob_enable_connection: float = 0.02
    prob_disable_connection: float = 0.01
    weight_mutation_power: float = 0.5
    weight_min: float = -0.1
    weight_max: float = 0.1
    mutate_weight_min: float = -5.0
    mutate_weight_max: float = 5.0
from collections import deque
import math
from typing import List, Dict, Set

from lab2.neat.genome import Genome
from lab2.neat.node_type import NodeType


def sigmoid(x: float) -> float:
    """
    Вычисляет модифицированную сигмоидную функцию активации, используемую в NEAT.
    Предотвращает переполнение экспоненты при больших отрицательных значениях.

    Args:
        x (float): Входное значение.

    Returns:
        float: Результат активации в диапазоне (0, 1).
    """
    if x >= 0:
        z = math.exp(-4.9 * x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(4.9 * x)
        return z / (1.0 + z)


class NeuralNetwork:
    """
    Нейронная сеть прямого распространения (Feed-Forward), построенная на основе
    структуры и весов генома NEAT с использованием топологической сортировки.
    """

    def __init__(self, genome: Genome) -> None:
        """
        Инициализирует нейронную сеть.

        Args:
            genome (Genome): Геном, задающий фенотип (структуру) сети.
        """
        self.genome: Genome = genome
        self.topological_order: List[int] = self._topological_sort()

    def _topological_sort(self) -> List[int]:
        """
        Выполняет топологическую сортировку узлов графа методом Кана.
        Используется для определения корректной последовательности вычисления сигналов.

        Returns:
            List[int]: Список идентификаторов узлов в порядке их активации.
        """
        indegree: Dict[int, int] = {node_id: 0 for node_id in self.genome.nodes}
        adjacency: Dict[int, List[int]] = {
            node_id: [] for node_id in self.genome.nodes
        }

        for conn in self.genome.connections.values():
            if not conn.is_enabled:
                continue
            adjacency[conn.in_node_id].append(conn.out_node_id)
            indegree[conn.out_node_id] += 1

        queue = deque(
            node_id for node_id, degree in indegree.items() if degree == 0
        )
        result: List[int] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbour in adjacency[node]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)

        return result

    def activate(self, inputs: List[float]) -> List[float]:
        """
        Выполняет прямой проход по нейронной сети по топологическому порядку.

        Args:
            inputs (List[float]): Список входных значений для сети.

        Returns:
            List[float]: Список выходных значений сети.

        Raises:
            ValueError: Если количество переданных входов не совпадает
                        с числом входных нейронов сети.
        """
        # Инициализация всех узлов нулями (включая изолированные узлы)
        values: Dict[int, float] = {node_id: 0.0 for node_id in self.genome.nodes}

        # Сортировка входных нейронов по ID и заполнение их внешними значениями
        input_nodes = sorted(
            (
                node
                for node in self.genome.nodes.values()
                if node.node_type == NodeType.INPUT
            ),
            key=lambda n: n.node_id,
        )

        if len(inputs) != len(input_nodes):
            raise ValueError(
                f"Input count mismatch: expected {len(input_nodes)}, got {len(inputs)}"
            )

        for node, value in zip(input_nodes, inputs):
            values[node.node_id] = value

        # Установка фиксированного значения для узлов смещения (Bias)
        for node in self.genome.nodes.values():
            if node.node_type == NodeType.BIAS:
                values[node.node_id] = 1.0

        # Узлы, значения которых заданы статически и не требуют суммирования входов
        fixed_ids: Set[int] = {
            node.node_id
            for node in self.genome.nodes.values()
            if node.node_type in (NodeType.INPUT, NodeType.BIAS)
        }

        # Вычисление взвешенных сумм и активация узлов по топологическому порядку
        for node_id in self.topological_order:
            if node_id in fixed_ids:
                continue

            total = sum(
                values.get(conn.in_node_id, 0.0) * conn.weight
                for conn in self.genome.connections.values()
                if conn.is_enabled and conn.out_node_id == node_id
            )
            values[node_id] = sigmoid(total)

        # Сбор значений с выходных узлов, отсортированных по их идентификаторам
        output_nodes = sorted(
            (
                node
                for node in self.genome.nodes.values()
                if node.node_type == NodeType.OUTPUT
            ),
            key=lambda n: n.node_id,
        )

        return [values.get(node.node_id, 0.0) for node in output_nodes]
import random
from typing import Dict, Any, Set

from lab2.neat.connection_gene import ConnectionGene
from lab2.neat.innovation_tracker import InnovationTracker, SplitRecord
from lab2.neat.node_gene import NodeGene
from lab2.neat.node_type import NodeType


class Genome:
    """
    Представляет геном особи в алгоритме NEAT, содержащий списки узлов и связей.
    """

    def __init__(self, genome_id: int) -> None:
        """
        Инициализирует новый экземпляр генома.

        Args:
            genome_id (int): Уникальный идентификатор генома.
        """
        self.genome_id: int = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0

    def clone(self) -> "Genome":
        """
        Создает глубокую копию текущего генома.

        Returns:
            Genome: Новый экземпляр генома с идентичной структурой и весами.
        """
        clone = Genome(self.genome_id)
        clone.fitness = self.fitness
        clone.adjusted_fitness = self.adjusted_fitness

        for node in self.nodes.values():
            clone.nodes[node.node_id] = node.clone()

        for conn in self.connections.values():
            clone.connections[conn.innovation_num] = conn.clone()

        return clone

    def mutate_weights(self, config: dict) -> None:
        """
        Выполняет мутацию весов существующих соединений (пертурбация или полная замена).

        Args:
            config (dict): Словарь с конфигурационными параметрами мутации.
        """
        for conn in self.connections.values():
            if random.random() > config["prob_mutate_weight"]:
                continue

            if random.random() < config["prob_weight_replace"]:
                # Полная замена веса в заданном диапазоне
                conn.weight = random.uniform(
                    config["mutate_weight_min"], config["mutate_weight_max"]
                )
            else:
                # Пертурбация: добавление небольшого гауссова шума
                conn.weight += random.gauss(0, config["weight_mutation_power"])
                conn.weight = max(
                    config["mutate_weight_min"],
                    min(config["mutate_weight_max"], conn.weight)
                )

    def __would_create_cycle(self, source_id: int, target_id: int) -> bool:
        """
        Проверяет, приведет ли добавление связи от source_id к target_id к появлению цикла.

        Args:
            source_id (int): Идентификатор начального узла.
            target_id (int): Идентификатор конечного узла.

        Returns:
            bool: True, если связь образует цикл, иначе False.
        """
        stack = [target_id]
        visited = set()

        while stack:
            current = stack.pop()

            if current == source_id:
                return True

            if current in visited:
                continue

            visited.add(current)

            for conn in self.connections.values():
                if not conn.is_enabled:
                    continue
                if conn.in_node_id == current:
                    stack.append(conn.out_node_id)

        return False

    def mutate_add_connection(
        self, tracker: InnovationTracker, config: dict
    ) -> None:
        """
        Пытается добавить новую случайную связь между существующими узлами генома.

        Args:
            tracker (InnovationTracker): Глобальный трекер инноваций алгоритма.
            config (dict): Словарь с конфигурационными параметрами.
        """
        if not self.nodes or random.random() > config["prob_add_conn"]:
            return

        all_node_ids = list(self.nodes.keys())

        for _ in range(100):
            in_node_id = random.choice(all_node_ids)
            out_node_id = random.choice(all_node_ids)

            if in_node_id == out_node_id:
                continue

            # Входные узлы и узлы смещения не могут быть целевыми
            if self.nodes[out_node_id].node_type in (NodeType.INPUT, NodeType.BIAS):
                continue

            # Проверка на существование такой связи
            exists = any(
                c.in_node_id == in_node_id and c.out_node_id == out_node_id
                for c in self.connections.values()
            )
            if exists:
                continue

            if self.__would_create_cycle(in_node_id, out_node_id):
                continue

            innovation_num = tracker.get_innovation_id(in_node_id, out_node_id)

            self.connections[innovation_num] = ConnectionGene(
                in_node_id=in_node_id,
                out_node_id=out_node_id,
                weight=random.uniform(config["weight_min"], config["weight_max"]),
                is_enabled=True,
                innovation_num=innovation_num,
            )
            return

    def mutate_add_node(
        self, tracker: InnovationTracker, node_tracker: Any, config: dict
    ) -> None:
        """
        Пытается добавить новый скрытый узел путем разделения случайной активной связи.

        Args:
            tracker (InnovationTracker): Глобальный трекер инноваций для связей.
            node_tracker (Any): Генератор уникальных идентификаторов для узлов.
            config (dict): Словарь с конфигурационными параметрами.
        """
        if not self.connections or random.random() > config["prob_add_node"]:
            return

        enabled_connections = [
            c for c in self.connections.values() if c.is_enabled
        ]
        if not enabled_connections:
            return

        old_connection = random.choice(enabled_connections)
        old_connection.is_enabled = False

        split_record = tracker.get_split_record(old_connection.innovation_num)

        if split_record is not None:
            new_node_id = split_record.new_node_id
            innovation1 = split_record.innovation1
            innovation2 = split_record.innovation2
        else:
            new_node_id = node_tracker.next_id()
            innovation1 = tracker.get_innovation_id(
                old_connection.in_node_id, new_node_id
            )
            innovation2 = tracker.get_innovation_id(
                new_node_id, old_connection.out_node_id
            )
            tracker.register_split(
                old_connection.innovation_num,
                SplitRecord(
                    new_node_id=new_node_id,
                    innovation1=innovation1,
                    innovation2=innovation2,
                ),
            )

        if new_node_id not in self.nodes:
            self.nodes[new_node_id] = NodeGene(
                node_id=new_node_id,
                node_type=NodeType.HIDDEN,
            )

        # Связь от старого входа к новому узлу с весом 1.0
        self.connections[innovation1] = ConnectionGene(
            in_node_id=old_connection.in_node_id,
            out_node_id=new_node_id,
            weight=1.0,
            is_enabled=True,
            innovation_num=innovation1,
        )
        # Связь от нового узла к старому выходу с исходным весом связи
        self.connections[innovation2] = ConnectionGene(
            in_node_id=new_node_id,
            out_node_id=old_connection.out_node_id,
            weight=old_connection.weight,
            is_enabled=True,
            innovation_num=innovation2,
        )

    @staticmethod
    def crossover(parent1: "Genome", parent2: "Genome", child_id: int) -> "Genome":
        """
        Выполняет кроссовер (скрещивание) двух родительских геномов.

        Args:
            parent1 (Genome): Первый родительский геном.
            parent2 (Genome): Второй родительский геном.
            child_id (int): Идентификатор создаваемого потомка.

        Returns:
            Genome: Новый геном потомка.
        """
        child = Genome(child_id)

        # Определение более приспособленного родителя
        fitter = None
        if parent1.fitness > parent2.fitness:
            fitter = parent1
        elif parent2.fitness > parent1.fitness:
            fitter = parent2

        innovations = set(parent1.connections.keys()) | set(parent2.connections.keys())

        for innovation in innovations:
            in_p1 = innovation in parent1.connections
            in_p2 = innovation in parent2.connections

            if in_p1 and in_p2:
                # Совпадающий ген: выбирается случайно у одного из родителей
                gene = random.choice(
                    [
                        parent1.connections[innovation],
                        parent2.connections[innovation],
                    ]
                )
                enabled = gene.is_enabled

                if (
                    not parent1.connections[innovation].is_enabled
                    or not parent2.connections[innovation].is_enabled
                ):
                    if random.random() < 0.75:
                        enabled = False

                copied = gene.clone()
                copied.is_enabled = enabled
                child.connections[innovation] = copied

            else:
                # Несовпадающий (disjoint/excess) ген
                if fitter is None:
                    gene = (
                        parent1.connections[innovation]
                        if in_p1
                        else parent2.connections[innovation]
                    )
                else:
                    if innovation not in fitter.connections:
                        continue
                    gene = fitter.connections[innovation]

                child.connections[innovation] = gene.clone()

        # Перенос структурных базовых узлов (INPUT, BIAS, OUTPUT)
        structural_types = (NodeType.INPUT, NodeType.BIAS, NodeType.OUTPUT)

        for node in parent1.nodes.values():
            if node.node_type in structural_types:
                child.nodes[node.node_id] = node.clone()

        for node in parent2.nodes.values():
            if node.node_type in structural_types and node.node_id not in child.nodes:
                child.nodes[node.node_id] = node.clone()

        # Сбор пула скрытых узлов для последующей фильтрации
        hidden_pool: Dict[int, NodeGene] = {}
        for node in parent1.nodes.values():
            if node.node_type == NodeType.HIDDEN:
                hidden_pool[node.node_id] = node
        for node in parent2.nodes.values():
            if node.node_type == NodeType.HIDDEN and node.node_id not in hidden_pool:
                hidden_pool[node.node_id] = node

        # Включение только тех скрытых узлов, которые задействованы в связях потомка
        needed: Set[int] = set()
        for conn in child.connections.values():
            if conn.in_node_id in hidden_pool:
                needed.add(conn.in_node_id)
            if conn.out_node_id in hidden_pool:
                needed.add(conn.out_node_id)

        for node_id in needed:
            child.nodes[node_id] = hidden_pool[node_id].clone()

        return child

    def mutate_disable_connection(self, probability: float) -> None:
        """
        С заданной вероятностью отключает случайную активную связь генома.

        Args:
            probability (float): Вероятность активации мутации отключения.
        """
        if random.random() > probability:
            return

        enabled = [c for c in self.connections.values() if c.is_enabled]
        if not enabled:
            return

        random.choice(enabled).is_enabled = False

    def mutate_enable_connection(self, probability: float) -> None:
        """
        С заданной вероятностью включает случайную неактивную связь генома.

        Args:
            probability (float): Вероятность активации мутации включения.
        """
        if random.random() > probability:
            return

        disabled = [c for c in self.connections.values() if not c.is_enabled]
        if not disabled:
            return

        random.choice(disabled).is_enabled = True
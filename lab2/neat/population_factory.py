import random
from typing import List, Any

from lab2.neat.connection_gene import ConnectionGene
from lab2.neat.genome import Genome
from lab2.neat.node_gene import NodeGene
from lab2.neat.node_type import NodeType


class PopulationFactory:
    """
    Фабрика для генерации популяций и начальных геномов в алгоритме NEAT.
    """

    @staticmethod
    def create_initial_population(
        population_size: int,
        num_inputs: int,
        num_outputs: int,
        innovation_tracker: Any,
    ) -> List[Genome]:
        """
        Создает начальную популяцию геномов со стандартной полносвязанной структурой.

        Каждый геном инициализируется заданным числом входных узлов, одним узлом
        смещения (Bias) и заданным числом выходных узлов. Все входные узлы и узел
        смещения соединяются со всеми выходными узлами со случайными весами в диапазоне [-1.0, 1.0].

        Args:
            population_size (int): Размер создаваемой популяции (количество особей).
            num_inputs (int): Количество входных нейронов.
            num_outputs (int): Количество выходных нейронов.
            innovation_tracker (Any): Глобальный трекер инноваций для регистрации связей.

        Returns:
            List[Genome]: Список сгенерированных геномов для начального поколения.
        """
        population: List[Genome] = []

        for genome_id in range(population_size):
            genome = Genome(genome_id)
            node_id = 0

            # Создание входных узлов
            input_ids = []
            for _ in range(num_inputs):
                genome.nodes[node_id] = NodeGene(node_id, NodeType.INPUT)
                input_ids.append(node_id)
                node_id += 1

            # Создание узла смещения (Bias)
            bias_id = node_id
            genome.nodes[bias_id] = NodeGene(bias_id, NodeType.BIAS)
            node_id += 1

            # Создание выходных узлов
            output_ids = []
            for _ in range(num_outputs):
                genome.nodes[node_id] = NodeGene(node_id, NodeType.OUTPUT)
                output_ids.append(node_id)
                node_id += 1

            # Создание полносвязанной структуры (входы и bias -> выходы)
            for input_id in input_ids + [bias_id]:
                for output_id in output_ids:
                    innovation = innovation_tracker.get_innovation_id(
                        input_id, output_id
                    )

                    genome.connections[innovation] = ConnectionGene(
                        in_node_id=input_id,
                        out_node_id=output_id,
                        weight=random.uniform(-1.0, 1.0),
                        is_enabled=True,
                        innovation_num=innovation,
                    )

            population.append(genome)

        return population
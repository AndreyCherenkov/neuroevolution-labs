import random
from typing import List

from lab2.neat.genome import Genome


def tournament_selection(genomes: List[Genome], tournament_size: int) -> Genome:
    """
    Выполняет турнирный отбор (Tournament Selection) среди заданных геномов.

    Случайным образом выбирает группу кандидатов указанного размера и возвращает
    особь с наибольшим значением приспособленности (fitness). Если размер списка
    особей меньше размера турнира, в группу кандидатов отбираются все доступные геномы.

    Args:
        genomes (List[Genome]): Список геномов, из которых производится выбор.
        tournament_size (int): Количество особей, случайным образом выбираемых для турнира.

    Returns:
        Genome: Победитель турнира (геном с максимальной приспособленностью).
    """
    candidates = random.sample(genomes, min(tournament_size, len(genomes)))
    return max(candidates, key=lambda g: g.fitness)
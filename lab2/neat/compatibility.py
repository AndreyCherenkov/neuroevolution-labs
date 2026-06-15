from lab2.neat.genome import Genome


def compatibility_distance(
        genome1: Genome,
        genome2: Genome,
        c1: float,
        c2: float,
        c3: float,
) -> float:
    """
    Вычисляет расстояние совместимости между двумя геномами по формуле NEAT.

    Параметры:
        genome1: Первый геном.
        genome2: Второй геном.
        c1: Коэффициент для избыточных (excess) генов.
        c2: Коэффициент для несоответствующих (disjoint) генов.
        c3: Коэффициент для средней разницы весов совпадающих генов.

    Возвращает:
        Значение расстояния совместимости.
    """
    innovations1 = set(genome1.connections.keys())
    innovations2 = set(genome2.connections.keys())

    matching = innovations1 & innovations2

    excess = 0
    disjoint = 0

    if not innovations1 or not innovations2:
        excess = len(innovations1 ^ innovations2)
    else:
        threshold = min(max(innovations1), max(innovations2))

        for innovation in innovations1 ^ innovations2:
            if innovation > threshold:
                excess += 1
            else:
                disjoint += 1

    if matching:
        avg_weight_diff = sum(
            abs(genome1.connections[innovation].weight - genome2.connections[innovation].weight)
            for innovation in matching
        ) / len(matching)
    else:
        avg_weight_diff = 0.0

    n = max(len(genome1.connections), len(genome2.connections), 1)

    return (
            c1 * excess / n
            + c2 * disjoint / n
            + c3 * avg_weight_diff
    )

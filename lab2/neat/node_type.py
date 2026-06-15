from enum import Enum


class NodeType(Enum):
    """
    Типы узлов (нейронов) в структуре сети NEAT.

    Attributes:
        INPUT: Входной узел для приема внешних сигналов.
        HIDDEN: Скрытый промежуточный узел, участвующий в вычислениях.
        OUTPUT: Выходной узел, возвращающий результат работы сети.
        BIAS: Узел смещения.
    """

    INPUT = "INPUT"
    HIDDEN = "HIDDEN"
    OUTPUT = "OUTPUT"
    BIAS = "BIAS"
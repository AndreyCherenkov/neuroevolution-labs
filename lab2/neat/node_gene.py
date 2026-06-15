from lab2.neat.node_type import NodeType


class NodeGene:
    """
    Представляет ген узла (нейрона) в структуре генома NEAT.

    Attributes:
        node_id (int): Уникальный идентификатор узла.
        node_type (NodeType): Тип узла (например, INPUT, HIDDEN, OUTPUT, BIAS).
        activation (str): Название функции активации, применяемой к узлу.
    """

    def __init__(
        self, node_id: int, node_type: NodeType, activation: str = "sigmoid"
    ) -> None:
        """
        Инициализирует новый экземпляр гена узла.

        Args:
            node_id (int): Уникальный идентификатор узла.
            node_type (NodeType): Тип узла.
            activation (str): Функция активации. По умолчанию "sigmoid".
        """
        self.node_id: int = node_id
        self.node_type: NodeType = node_type
        self.activation: str = activation

    def clone(self) -> "NodeGene":
        """
        Создает копию текущего гена узла.

        Returns:
            NodeGene: Новый экземпляр гена узла с теми же параметрами.
        """
        return NodeGene(
            node_id=self.node_id,
            node_type=self.node_type,
            activation=self.activation,
        )
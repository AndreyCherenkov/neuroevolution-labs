class ConnectionGene:
    """
    Ген связи между двумя нейронами в геноме NEAT.

    Атрибуты:
        in_node_id: Идентификатор входного нейрона.
        out_node_id: Идентификатор выходного нейрона.
        weight: Вес соединения.
        is_enabled: Флаг активности соединения.
        innovation_num: Уникальный номер инновации.
    """

    def __init__(
        self,
        in_node_id: int,
        out_node_id: int,
        weight: float,
        is_enabled: bool,
        innovation_num: int,
    ) -> None:
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.is_enabled = is_enabled
        self.innovation_num = innovation_num

    def key(self) -> tuple[int, int]:
        """
        Возвращает уникальный ключ соединения.

        Returns:
            Кортеж из идентификаторов входного и выходного нейронов.
        """
        return self.in_node_id, self.out_node_id

    def clone(self) -> "ConnectionGene":
        """
        Создает полную копию гена связи.

        Returns:
            Новый объект ConnectionGene с теми же параметрами.
        """
        return ConnectionGene(
            in_node_id=self.in_node_id,
            out_node_id=self.out_node_id,
            weight=self.weight,
            is_enabled=self.is_enabled,
            innovation_num=self.innovation_num,
        )
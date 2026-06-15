class NodeIdTracker:
    """
    Генератор уникальных последовательных идентификаторов для узлов нейронной сети.
    """

    def __init__(self, start_id: int = 0) -> None:
        """
        Инициализирует трекер идентификаторов.

        Args:
            start_id (int): Начальное значение счетчика. По умолчанию 0.
        """
        self.current_id: int = start_id

    def next_id(self) -> int:
        """
        Инкрементирует счетчик и возвращает следующий уникальный идентификатор.

        Returns:
            int: Новый уникальный идентификатор узла.
        """
        self.current_id += 1
        return self.current_id

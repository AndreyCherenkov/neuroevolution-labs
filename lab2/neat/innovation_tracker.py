from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class SplitRecord:
    """
    Запись о разделении существующей связи при добавлении нового узла.

    Attributes:
        new_node_id (int): Идентификатор вновь созданного скрытого узла.
        innovation1 (int): Номер инновации для связи от исходного входа к новому узлу.
        innovation2 (int): Номер инновации для связи от нового узла к исходному выходу.
    """

    new_node_id: int
    innovation1: int
    innovation2: int


class InnovationTracker:
    """
    Глобальный трекер инноваций для алгоритма NEAT.
    Используется для отслеживания появления новых связей и узлов во избежание
    дублирования номеров инноваций при одинаковых структурных мутациях.
    """

    def __init__(self) -> None:
        """Инициализирует трекер со сброшенными счетчиками и пустой историей."""
        self.current_innovation: int = 0
        self.connection_history: Dict[Tuple[int, int], int] = {}
        self.split_history: Dict[int, SplitRecord] = {}

    def get_innovation_id(self, in_node_id: int, out_node_id: int) -> int:
        """
        Возвращает существующий ID инновации для связи или генерирует новый,
        если такая связь ранее не создавалась в текущем поколении.

        Args:
            in_node_id (int): Идентификатор начального узла.
            out_node_id (int): Идентификатор конечного узла.

        Returns:
            int: Уникальный идентификатор инновации для данной связи.
        """
        edge = (in_node_id, out_node_id)

        if edge in self.connection_history:
            return self.connection_history[edge]

        self.current_innovation += 1
        innovation = self.current_innovation
        self.connection_history[edge] = innovation

        return innovation

    def get_split_record(self, old_connection_innovation: int) -> Optional[SplitRecord]:
        """
        Возвращает информацию о разделении связи, если оно уже выполнялось ранее.

        Args:
            old_connection_innovation (int): Инновационный номер разделяемой связи.

        Returns:
            Optional[SplitRecord]: Структура с данными разделения или None,
                                  если эта связь разделяется впервые.
        """
        return self.split_history.get(old_connection_innovation)

    def register_split(
        self, old_connection_innovation: int, record: SplitRecord
    ) -> None:
        """
        Регистрирует факт разделения связи в глобальной истории трекера.

        Args:
            old_connection_innovation (int): Инновационный номер разделяемой связи.
            record (SplitRecord): Объект с информацией о результатах разделения.
        """
        self.split_history[old_connection_innovation] = record
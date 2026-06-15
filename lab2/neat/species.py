from lab2.neat.genome import Genome


class Species:
    """
    Представляет отдельный биологический вид (подпопуляцию) в алгоритме NEAT.

    Объединяет схожие геномы для защиты инновационных структур от преждевременного
    вытеснения и управляет механизмами стагнации и корректировки приспособленности.

    Attributes:
        species_id (int): Уникальный идентификатор вида.
        representative (Genome): Геном-представитель, используемый для вычисления
            расстояния совместимости с новыми особями.
        members (list[Genome]): Список геномов, входящих в данный вид.
        best_fitness (float): Максимальное значение приспособленности, достигнутое
            в этом виде за всю историю его существования.
        staleness (int): Количество поколений подряд, в которых вид не улучшал
            свой максимальный показатель приспособленности.
    """

    def __init__(self, species_id: int, representative: Genome) -> None:
        """
        Инициализирует новый экземпляр вида.

        Args:
            species_id (int): Уникальный идентификатор вида.
            representative (Genome): Геном, который станет первым эталоном вида.
        """
        self.species_id: int = species_id
        self.representative: Genome = representative.clone()
        self.members: list[Genome] = []
        self.best_fitness: float = float("-inf")
        self.staleness: int = 0

    def add_member(self, genome: Genome) -> None:
        """
        Добавляет новый геном в состав данного вида.

        Args:
            genome (Genome): Добавляемый геном особи.
        """
        self.members.append(genome)

    def clear_members(self) -> None:
        """Очищает список членов вида перед распределением нового поколения."""
        self.members.clear()

    def sort_members(self) -> None:
        """Сортирует геномы внутри вида по убыванию их приспособленности."""
        self.members.sort(key=lambda g: g.fitness, reverse=True)

    @property
    def champion(self) -> Genome:
        """
        Возвращает лучшую особь (чемпиона) в текущем составе вида.

        Returns:
            Genome: Геном с наибольшей приспособленностью.
        """
        self.sort_members()
        return self.members[0]

    def update_staleness(self) -> None:
        """
        Обновляет счетчик поколений без улучшений (стагнации).
        Если лучший результат текущего поколения превосходит исторический максимум вида,
        счетчик сбрасывается в 0, иначе увеличивается на 1.
        """
        if not self.members:
            return

        self.sort_members()
        current_best = self.members[0].fitness

        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.staleness = 0
        else:
            self.staleness += 1

    def compute_adjusted_fitness(self) -> None:
        """
        Вычисляет скорректированную приспособленность (adjusted fitness) для каждого
        члена вида. Значение fitness каждого генома делится на общий размер вида,
        что реализует механизм разделения экологической ниши (fitness sharing).
        """
        size = len(self.members)
        if size == 0:
            return

        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / size

    def total_adjusted_fitness(self) -> float:
        """
        Вычисляет сумму скорректированных приспособленностей всех особей вида.
        Используется для пропорционального распределения квоты потомков.

        Returns:
            float: Суммарная скорректированная приспособленность вида.
        """
        return sum(g.adjusted_fitness for g in self.members)

    def get_elites(self, n: int) -> list[Genome]:
        """
        Возвращает n лучших особей вида.

        Args:
            n (int): Количество запрашиваемых элитных особей.

        Returns:
            list[Genome]: Список из n наиболее приспособленных геномов вида.
        """
        self.sort_members()
        return self.members[:n]
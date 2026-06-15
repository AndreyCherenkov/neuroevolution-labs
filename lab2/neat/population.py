import random
from typing import List, Any, Optional, Tuple

from lab2.neat.compatibility import compatibility_distance
from lab2.neat.genome import Genome
from lab2.neat.selection import tournament_selection
from lab2.neat.species import Species


class Population:
    """
    Управляет популяцией организмов (геномов) в алгоритме NEAT.
    Отвечает за видообразование, оценку приспособленности, селекцию и репродукцию.
    """

    def __init__(
        self,
        genomes: List[Genome],
        config: Any,
        innovation_tracker: Any,
        node_tracker: Any,
    ) -> None:
        """
        Инициализирует новый экземпляр популяции.

        Args:
            genomes (List[Genome]): Начальный список геномов популяции.
            config (Any): Конфигурация параметров NEATConfig.
            innovation_tracker (Any): Глобальный трекер инноваций для связей.
            node_tracker (Any): Трекер идентификаторов для новых узлов.
        """
        self.genomes: List[Genome] = genomes
        self.config: Any = config
        self.innovation_tracker: Any = innovation_tracker
        self.node_tracker: Any = node_tracker
        self.species: List[Species] = []
        self.generation: int = 0
        self.best_genome: Optional[Genome] = None

    def speciate(self) -> None:
        """
        Разделяет геномы текущего поколения по видам (Species)
        на основе их генетического расстояния совместимости.
        """
        for sp in self.species:
            if sp.members:
                sp.representative = random.choice(sp.members).clone()
            sp.clear_members()

        for genome in self.genomes:
            assigned = False

            for sp in self.species:
                distance = compatibility_distance(
                    genome,
                    sp.representative,
                    self.config.c1,
                    self.config.c2,
                    self.config.c3,
                )

                if distance < self.config.compatibility_threshold:
                    sp.add_member(genome)
                    assigned = True
                    break

            if not assigned:
                new_species = Species(len(self.species), genome)
                new_species.add_member(genome)
                self.species.append(new_species)

        # Удаление пустых видов
        self.species = [s for s in self.species if len(s.members) > 0]

    def remove_stale_species(self) -> None:
        """
        Удаляет стагнирующие виды, которые не улучшали свои результаты
        в течение заданного числа поколений. Главный чемпион популяции защищен от удаления.
        """
        if not self.species:
            return

        for sp in self.species:
            sp.update_staleness()

        best_species = max(self.species, key=lambda s: s.champion.fitness)

        survivors = []
        for sp in self.species:
            if sp.staleness < self.config.max_stagnation or sp == best_species:
                if sp == best_species and sp.staleness >= self.config.max_stagnation:
                    sp.staleness = 0
                survivors.append(sp)

        self.species = survivors

    def compute_adjusted_fitness(self) -> None:
        """Вычисляет скорректированную приспособленность для каждого вида."""
        for sp in self.species:
            sp.compute_adjusted_fitness()

    def update_best_genome(self) -> None:
        """Обновляет глобальный лучший геном за всю историю эволюции."""
        current_best = max(self.genomes, key=lambda g: g.fitness)

        if (
            self.best_genome is None
            or current_best.fitness > self.best_genome.fitness
        ):
            self.best_genome = current_best.clone()

    def reproduce(self) -> None:
        """
        Формирует следующее поколение популяции посредством элитизма,
        турнирного отбора, кроссовера и последующих мутаций потомков.
        """
        next_generation: List[Genome] = []

        total_adjusted_fitness = sum(
            sp.total_adjusted_fitness() for sp in self.species
        )

        # Расчет квот и элит для каждого вида
        species_quotas: List[Tuple[Species, int, int]] = []
        for sp in self.species:
            sp.sort_members()

            # Определение количества элит на основе размера вида
            if len(sp.members) < 5:
                n_elites = 1
            else:
                n_elites = max(
                    1, round(len(sp.members) * self.config.elite_fraction)
                )

            # Вычисление общего количества потомков для вида
            if total_adjusted_fitness == 0:
                offspring_count = self.config.population_size // len(self.species)
            else:
                offspring_count = max(
                    n_elites,
                    round(
                        sp.total_adjusted_fitness()
                        / total_adjusted_fitness
                        * self.config.population_size
                    ),
                )

            species_quotas.append((sp, n_elites, offspring_count))

        # Заполнение следующего поколения организмами
        for sp, n_elites, offspring_count in species_quotas:
            # Копирование элитных особей без изменений
            for elite in sp.members[:n_elites]:
                next_generation.append(elite.clone())

            # Создание потомков через кроссовер и мутации
            n_offspring = offspring_count - n_elites
            for _ in range(max(0, n_offspring)):
                parent1 = tournament_selection(
                    sp.members, self.config.tournament_size
                )
                parent2 = tournament_selection(
                    sp.members, self.config.tournament_size
                )

                child = Genome.crossover(
                    parent1, parent2, random.randint(0, 1_000_000)
                )

                # Применение операторов мутации к новому организму
                config_dict = vars(self.config)
                child.mutate_weights(config_dict)
                child.mutate_add_connection(self.innovation_tracker, config_dict)
                child.mutate_add_node(
                    self.innovation_tracker, self.node_tracker, config_dict
                )
                child.mutate_enable_connection(self.config.prob_enable_connection)
                child.mutate_disable_connection(self.config.prob_disable_connection)

                next_generation.append(child)

        # Корректировка размера популяции из-за погрешностей округления квот
        while len(next_generation) < self.config.population_size:
            next_generation.append(random.choice(next_generation).clone())

        self.genomes = next_generation[: self.config.population_size]

    def evolve(self) -> None:
        """Выполняет полный цикл эволюции для одного поколения."""
        self.speciate()
        self.remove_stale_species()
        self.compute_adjusted_fitness()
        self.update_best_genome()
        self.reproduce()
        self.generation += 1
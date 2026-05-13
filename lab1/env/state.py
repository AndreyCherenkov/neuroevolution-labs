from dataclasses import dataclass

import numpy as np


@dataclass
class DiscreteState:
    delta_x: int
    delta_y: int
    battery: int
    velocity: int

    def to_index(self) -> int:
        return (
                self.delta_x * 14 * 12 * 4 +
                self.delta_y * 12 * 4 +
                self.battery * 4 +
                self.velocity
        )


@dataclass
class State:
    delta_x: float
    delta_y: float
    battery: float
    velocity: float

    @staticmethod
    def __discretize_value(
            value: float,
            min_value: float,
            max_value: float,
            states: int
    ) -> int:
        value = np.clip(value, min_value, max_value)

        step = (max_value - min_value) / states

        index = ((value - min_value) / step).astype(int)  # todo astype or int()?

        return min(index, states - 1)  # todo вовзаращть от 0 до max_state - 1 или от 1 до max_state?

    def discretize(self) -> DiscreteState:
        return DiscreteState(
            self.__discretize_value(self.delta_x, 0, 24, 14),
            self.__discretize_value(self.delta_y, 0, 12, 14),
            self.__discretize_value(self.battery, 0, 1, 12),
            self.__discretize_value(self.velocity, 0.10, 0.60, 4)
        )

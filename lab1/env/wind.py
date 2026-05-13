import random


class Wind:
    def __init__(
            self,
            ux: list[float] = (-0.11, 0.11),
            uy: list[float] = (-0.11, 0.11),
            update_rate_steps: int = 15
    ):
        self.ux = ux
        self.uy = uy

        self.wx = random.uniform(ux[0], ux[1])
        self.wy = random.uniform(uy[0], uy[1])

        self.update_rate_steps = update_rate_steps

    def current(self) -> tuple[float, float]:
        return self.wx, self.wy

    def update(self) -> tuple[float, float]:
        self.wx = random.uniform(self.ux[0], self.ux[1])
        self.wy = random.uniform(self.uy[0], self.uy[1])
        return self.wx, self.wy
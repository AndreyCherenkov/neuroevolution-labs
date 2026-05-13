import numpy as np


class AgentSpawn:
    def __init__(
            self,
            x: float = 3.5,
            y: float = 6.0,
            sigma: float = 1.8,
            clip_x: tuple[float, float] = (1.0, 7.0),
            clip_y: tuple[float, float] = (2.0, 10.0),
    ):
        pos = np.random.normal(
            loc=[x, y],
            scale=sigma,
            size=2,
        )

        pos[0] = np.clip(pos[0], clip_x[0], clip_x[1])
        pos[1] = np.clip(pos[1], clip_y[0], clip_y[1])

        self.x_spawn = pos[0]
        self.y_spawn = pos[1]
from argparse import Action
from dataclasses import dataclass
from enum import Enum

from numpy import pi


class Direction(Enum):
    RIGHT = 0.0
    UP = pi / 2
    LEFT = pi
    DOWN = 3 * pi / 2



@dataclass(frozen=True)
class Action:
    direction: Direction
    velocity: float

    @staticmethod
    def get_all_actions(velocities: list[float]) -> list[Action]:
        actions = []
        for direction in Direction:
            for velocity in velocities:
                actions.append(Action(direction, velocity))

        return actions

""""
Станция зарядки: центр (21.5,8.5), радиус 0.9, мощность Pcharge = 0.025 на шаг.

Что делает
расход энергии;
зарядка на станции;
ограничение диапазона;
проверка критического заряда.
"""


class RechargeStation:
    def __init__(
            self,
            x0: float = 21.5,
            y0: float = 8.5,
            radius: float = 0.9,
            power_charge: float = 0.025
    ):
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        self.power_charge = power_charge

    def contains(self, x, y) -> bool:
        dx = x - self.x0
        dy = y - self.y0
        return (dx ** 2 + dy ** 2) <= self.radius

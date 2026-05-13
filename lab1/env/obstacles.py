class Obstacle:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def contains(self, x, y) -> bool:
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)
from dataclasses import dataclass

@dataclass(frozen=True)
class TestMetric:
    success_percentage: float
    average_success_episode_length: float
    average_charge_cost: float
    average_episode_collision_percentage: float

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeLog:
    summary_reward: float
    episode_length: int
    collision_number: int
    final_charge: float

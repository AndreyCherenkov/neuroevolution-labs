from numpy import random, clip
from numpy import cos, sin

from lab1.env.action import Action
from lab1.env.agent_spawn import AgentSpawn
from lab1.env.obstacles import Obstacle
from lab1.env.recharge_station import RechargeStation
from lab1.env.state import State
from lab1.env.wind import Wind


class Environment:
    def __init__(
            self,
            x_bounds: tuple[float, float],
            y_bounds: tuple[float, float],
            station: RechargeStation,
            agent_spawn: AgentSpawn,
            wind: Wind,
            velocities: list[float] = (0.10, 0.25, 0.40, 0.60),
            obstacles: list[Obstacle] = (),
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.station = station
        self.agent_spawn = agent_spawn
        self.wind = wind
        self.velocities = velocities
        self.obstacles = obstacles
        self.state: State | None = None

        self.current_x = agent_spawn.x_spawn
        self.current_y = agent_spawn.y_spawn

        self.actions = Action.get_all_actions(velocities)

        self.time = 0

    def reset(self) -> State:
        self.agent_spawn = AgentSpawn()

        x_spawn = self.agent_spawn.x_spawn
        y_spawn = self.agent_spawn.y_spawn

        self.current_x = x_spawn
        self.current_y = y_spawn

        self.state = State(
            self.station.x0 - x_spawn,
            self.station.y0 - y_spawn,
            1.0,
            self.velocities[random.randint(0, len(self.velocities) - 1)],
        )

        self.time = 0

        return self.state

    def step(self, action_idx: int) -> (State, float, bool, bool, bool):
        if self.state is None:
            raise RuntimeError("Environment should be reset()")

        if self.time > 0 and self.time % self.wind.update_rate_steps == 0:
            self.wind.update()

        wind_x, wind_y = self.wind.current()

        action = self.actions[action_idx]
        dx = cos(action.direction.value) * action.velocity
        dy = sin(action.direction.value) * action.velocity

        new_x = self.current_x + dx + wind_x
        new_y = self.current_y + dy + wind_y

        new_x = clip(new_x, self.x_bounds[0], self.x_bounds[1])
        new_y = clip(new_y, self.y_bounds[0], self.y_bounds[1])

        # препятствия
        collided = False

        for obstacle in self.obstacles:
            if obstacle.contains(new_x, new_y):
                collided = True
                break

        if collided:
            new_x = self.current_x
            new_y = self.current_y

        # батарея
        battery = (self.state.battery - 0.018 * action.velocity ** 2 - 0.003 * (wind_x ** 2 + wind_y ** 2) -
                   0.002 * abs(action.velocity))

        in_station = self.station.contains(new_x, new_y)
        if in_station:
            battery += 0.025

        battery = min(1.0, max(0.0, battery))

        self.current_x = new_x
        self.current_y = new_y

        self.state = State(
            self.station.x0 - self.current_x,
            self.station.y0 - self.current_y,
            battery,
            action.velocity
        )

        self.time += 1

        return self.state, *self.__reward(self.state, in_station), collided

    def __reward(self, state: State, in_station: bool) -> (float, bool, bool):
        station = self.station

        success = (pow(pow(self.current_x - station.x0, 2) + pow(self.current_y - station.y0, 2), 0.5) < 0.9 and
                   state.battery > 0.90 and
                   self.time <= 140)
        failed = state.battery < 0.04 or self.time >= 200

        if success:
            return 90, success, True
        elif failed:
            return -170, failed, False
        else:
            reward = -1.0 - 0.030 * pow(state.velocity, 2)
            if in_station:
                reward += 0.12 * 0.025
            return reward, False, False

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state_idx: int) -> int:
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

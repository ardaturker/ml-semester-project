from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """Abstract base class — all environments must implement this contract."""

    @abstractmethod
    def reset(self):
        """Reset environment to initial state. Returns initial state (int or array)."""
        ...

    @abstractmethod
    def step(self, action: int):
        """
        Take an action.
        Returns: (next_state, reward, done)
        """
        ...

    @property
    @abstractmethod
    def action_space(self) -> int:
        """Number of discrete actions available."""
        ...

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Number of distinct states (for tabular) or state vector size (for DQN)."""
        ...

    def render(self):
        """Optional: print a visual representation of the current state."""
        pass

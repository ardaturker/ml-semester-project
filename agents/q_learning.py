import numpy as np
import pickle
from collections import defaultdict
import config


class QLearningAgent:
    """
    Tabular Q-Learning (off-policy TD control).

    Update rule:
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

    'Off-policy' means we update toward the greedy (best) next action,
    regardless of what action we actually take next. This is different from
    SARSA, which updates toward the action we actually took.
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

        # Q-table: state → array of Q-values per action
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Q-Learning update — bootstraps from max Q(next_state)."""
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "epsilon": self.epsilon}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Support both old format (dict only) and new format (dict with epsilon)
        if isinstance(data, dict) and "q_table" in data:
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), data["q_table"])
            self.epsilon = data.get("epsilon", self.epsilon_end)
        else:
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), data)

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import config


# ── Neural Network ────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Two-layer feedforward network that approximates Q(s, a)."""

    def __init__(self, state_size: int, action_size: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network (DQN) agent.

    Improvements over tabular Q-Learning:
    1. Neural network approximates Q(s,a) — scales to large/continuous state spaces
    2. Experience replay — breaks correlations between consecutive samples
    3. Target network — separate network for stable TD targets, synced every N steps

    Same off-policy update logic as Q-Learning, but:
        loss = MSE(Q(s,a),  r + γ * max_a' Q_target(s', a'))
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ

        self.device = torch.device("cpu")

        # Online network (trained every step) and target network (synced periodically)
        self.online_net = QNetwork(state_size, action_size, config.DQN_HIDDEN_SIZE).to(self.device)
        self.target_net = QNetwork(state_size, action_size, config.DQN_HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.DQN_LR)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        self.steps = 0

    def _state_tensor(self, state: int) -> torch.Tensor:
        """One-hot encode integer state for the network input."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return torch.tensor(vec, device=self.device).unsqueeze(0)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            q_vals = self.online_net(self._state_tensor(state))
        return int(q_vals.argmax().item())

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        self.replay.push(state, action, reward, next_state, done)
        self.steps += 1

        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Build tensors
        state_vecs = torch.zeros(self.batch_size, self.state_size, device=self.device)
        next_state_vecs = torch.zeros(self.batch_size, self.state_size, device=self.device)
        for i, (s, ns) in enumerate(zip(states, next_states)):
            state_vecs[i, s] = 1.0
            next_state_vecs[i, ns] = 1.0

        actions_t  = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t  = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t    = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values for taken actions
        q_values = self.online_net(state_vecs).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_vecs).max(1).values
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.steps   = ckpt.get("steps", 0)

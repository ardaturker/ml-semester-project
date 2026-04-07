# config.py — All hyperparameters in one place

# ── Training ──────────────────────────────────────────────
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500

# ── Q-Learning / SARSA ───────────────────────────────────
ALPHA = 0.1          # learning rate
GAMMA = 0.95         # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997   # slower decay for larger state space

# ── DQN ──────────────────────────────────────────────────
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 20000
TARGET_UPDATE_FREQ = 200
DQN_LR = 0.001
DQN_HIDDEN_SIZE = 128   # wider network for larger state space

# ── Environments ─────────────────────────────────────────
MAZE_SIZE = 12           # was 6 — 144 states
GRID_SIZE = 16           # was 8 — 256 states
GRID_OBSTACLE_DENSITY = 0.15
FEUDAL_MAX_TURNS = 50

# ── Live plot ─────────────────────────────────────────────
PLOT_UPDATE_EVERY = 10   # redraw live curve every N episodes

"""
train.py — Train RL agents with live learning curve + save/load support.

Usage:
    python3 train.py                              # train all 9 combos
    python3 train.py --env maze --agent qlearning # single combo
    python3 train.py --env maze --agent qlearning --continue  # resume saved agent
    python3 train.py --env maze --agent qlearning --episodes 500

Live plot: a matplotlib window opens and updates every 10 episodes.
Agents are auto-saved to results/agents/ after training.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

import config
from environments.maze_env import MazeEnv
from environments.grid_env import GridEnv
from environments.feudal_env import FeudalEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent


ENVS = {
    "maze":   lambda: MazeEnv(size=config.MAZE_SIZE),
    "grid":   lambda: GridEnv(size=config.GRID_SIZE, obstacle_density=config.GRID_OBSTACLE_DENSITY),
    "feudal": lambda: FeudalEnv(max_turns=config.FEUDAL_MAX_TURNS),
}

AGENTS = ["qlearning", "sarsa", "dqn"]

COLORS = {"qlearning": "#2196F3", "sarsa": "#FF9800", "dqn": "#4CAF50"}
LABELS = {"qlearning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN"}


# ── Agent factory ─────────────────────────────────────────────────────────────

def make_agent(name: str, env):
    if name == "qlearning": return QLearningAgent(env.state_size, env.action_space)
    if name == "sarsa":     return SARSAAgent(env.state_size, env.action_space)
    if name == "dqn":       return DQNAgent(env.state_size, env.action_space)
    raise ValueError(f"Unknown agent: {name}")


# ── Save / load paths ─────────────────────────────────────────────────────────

def agent_path(env_name: str, agent_name: str) -> str:
    os.makedirs("results/agents", exist_ok=True)
    ext = ".pt" if agent_name == "dqn" else ".pkl"
    return f"results/agents/{env_name}_{agent_name}{ext}"


def rewards_path(env_name: str, agent_name: str) -> str:
    return f"results/{env_name}_{agent_name}_rewards.csv"


def load_existing_rewards(env_name: str, agent_name: str) -> list:
    path = rewards_path(env_name, agent_name)
    if os.path.exists(path):
        return pd.read_csv(path)["reward"].tolist()
    return []


# ── Episode runners ───────────────────────────────────────────────────────────

def run_episode(env, agent, agent_name: str) -> float:
    state = env.reset()
    total = 0.0
    if agent_name == "sarsa":
        action = agent.select_action(state)
    for _ in range(config.MAX_STEPS_PER_EPISODE):
        if agent_name == "sarsa":
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
        else:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        total += reward
        if done:
            break
    return total


# ── Live plot ─────────────────────────────────────────────────────────────────

class LivePlot:
    """Matplotlib window that updates during training."""

    WINDOW = 30  # rolling average window

    def __init__(self, env_name: str, agent_name: str, total_episodes: int, offset: int = 0):
        self.env_name = env_name
        self.agent_name = agent_name
        self.total_episodes = total_episodes
        self.offset = offset  # episodes already trained before this run

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.canvas.manager.set_window_title(f"Training: {env_name} × {agent_name}")
        color = COLORS.get(agent_name, "#2196F3")
        self.line_raw,  = self.ax.plot([], [], color=color, alpha=0.25, linewidth=1, label="Raw reward")
        self.line_avg,  = self.ax.plot([], [], color=color, linewidth=2.5, label=f"Rolling avg ({self.WINDOW} ep)")
        self.ax.set_xlabel("Episode (total across all runs)")
        self.ax.set_ylabel("Episode Reward")
        self.ax.set_title(f"Live Learning Curve — {env_name.capitalize()} | {LABELS.get(agent_name, agent_name)}")
        self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self._info = self.ax.text(0.02, 0.96, "", transform=self.ax.transAxes,
                                   fontsize=9, verticalalignment="top",
                                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, all_rewards: list, epsilon: float):
        episodes = list(range(1, len(all_rewards) + 1))
        smoothed = pd.Series(all_rewards).rolling(self.WINDOW, min_periods=1).mean().tolist()

        self.line_raw.set_data(episodes, all_rewards)
        self.line_avg.set_data(episodes, smoothed)
        self.ax.relim(); self.ax.autoscale_view()

        avg50 = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
        best  = max(all_rewards)
        new_ep = len(all_rewards) - self.offset
        self._info.set_text(
            f"This run: {new_ep}/{self.total_episodes} ep  |  "
            f"ε={epsilon:.3f}  |  avg(last 50)={avg50:.1f}  |  best={best:.0f}"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()


# ── Main training function ────────────────────────────────────────────────────

def train(env_name: str, agent_name: str, episodes: int, resume: bool = False,
          live_plot: bool = True) -> list:
    env = ENVS[env_name]()
    agent = make_agent(agent_name, env)

    # Load existing agent and rewards if resuming
    existing_rewards = []
    ap = agent_path(env_name, agent_name)
    if resume and os.path.exists(ap):
        agent.load(ap)
        existing_rewards = load_existing_rewards(env_name, agent_name)
        print(f"  Loaded agent from {ap}  (ε={agent.epsilon:.3f}, {len(existing_rewards)} episodes so far)")
    elif resume:
        print(f"  No saved agent found at {ap} — starting fresh.")

    all_rewards = list(existing_rewards)
    offset = len(all_rewards)

    plot = LivePlot(env_name, agent_name, episodes, offset=offset) if live_plot else None

    print(f"  Training for {episodes} episodes  (ε starts at {agent.epsilon:.3f})\n")

    for ep in range(1, episodes + 1):
        r = run_episode(env, agent, agent_name)
        agent.decay_epsilon()
        all_rewards.append(r)

        if ep % 50 == 0:
            avg = np.mean(all_rewards[-50:])
            print(f"  ep {ep:5d}/{episodes}  |  avg(last 50)={avg:8.1f}  |  ε={agent.epsilon:.4f}")

        if plot and ep % config.PLOT_UPDATE_EVERY == 0:
            plot.update(all_rewards, agent.epsilon)

    if plot:
        plot.update(all_rewards, agent.epsilon)
        plot.close()

    # Auto-save agent
    agent.save(ap)
    print(f"\n  Agent saved → {ap}")

    return all_rewards


def save_rewards(env_name: str, agent_name: str, rewards: list):
    os.makedirs("results", exist_ok=True)
    path = rewards_path(env_name, agent_name)
    pd.DataFrame({"episode": range(1, len(rewards) + 1), "reward": rewards}).to_csv(path, index=False)
    print(f"  Rewards saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RL agents with live learning curve.")
    parser.add_argument("--env",      choices=list(ENVS.keys()), default=None)
    parser.add_argument("--agent",    choices=AGENTS,            default=None)
    parser.add_argument("--episodes", type=int, default=config.EPISODES,
                        help=f"Episodes to train (default: {config.EPISODES})")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Load saved agent and continue training from where it left off")
    parser.add_argument("--no-plot",  dest="no_plot", action="store_true",
                        help="Disable live plot window")
    args = parser.parse_args()

    env_names   = [args.env]   if args.env   else list(ENVS.keys())
    agent_names = [args.agent] if args.agent else AGENTS

    combos = len(env_names) * len(agent_names)
    mode   = "continuing" if args.resume else "fresh"
    print(f"\nTraining {combos} combo(s) | {args.episodes} episodes each | mode: {mode}\n")

    for env_name in env_names:
        for agent_name in agent_names:
            print(f"▶ {env_name} × {agent_name}")
            rewards = train(env_name, agent_name, episodes=args.episodes,
                            resume=args.resume, live_plot=not args.no_plot)
            save_rewards(env_name, agent_name, rewards)
            print()

    print("Done. Run `python3 visualize.py` to regenerate static plots.\n")


if __name__ == "__main__":
    main()

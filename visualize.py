"""
visualize.py — Generate all plots from saved training results.

Run after train.py:
    python visualize.py

Outputs to results/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

ENVS   = ["maze", "grid", "feudal"]
AGENTS = ["qlearning", "sarsa", "dqn"]
COLORS = {"qlearning": "#2196F3", "sarsa": "#FF9800", "dqn": "#4CAF50"}
LABELS = {"qlearning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN"}
WINDOW = 20  # rolling average window


def load(env_name: str, agent_name: str) -> pd.DataFrame | None:
    path = f"results/{env_name}_{agent_name}_rewards.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def smooth(values, window: int = WINDOW) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().values


# ── 1. Learning curves — one plot per environment ─────────────────────────────

def plot_learning_curves():
    os.makedirs("results/figures", exist_ok=True)
    for env_name in ENVS:
        fig, ax = plt.subplots(figsize=(9, 5))
        found = False
        for agent_name in AGENTS:
            df = load(env_name, agent_name)
            if df is None:
                continue
            found = True
            smoothed = smooth(df["reward"].values)
            ax.plot(df["episode"], smoothed, color=COLORS[agent_name],
                    label=LABELS[agent_name], linewidth=2)
            ax.fill_between(df["episode"],
                            pd.Series(df["reward"]).rolling(WINDOW, min_periods=1).min(),
                            pd.Series(df["reward"]).rolling(WINDOW, min_periods=1).max(),
                            alpha=0.1, color=COLORS[agent_name])

        if not found:
            plt.close()
            continue

        ax.set_title(f"Learning Curves — {env_name.capitalize()} Environment", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Total Reward (smoothed, window={WINDOW})")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        path = f"results/figures/learning_curves_{env_name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved → {path}")


# ── 2. Policy heatmap — Maze (Q-Learning) ────────────────────────────────────

def plot_maze_policy():
    """Show the learned greedy policy on the maze grid."""
    import config
    from environments.maze_env import MazeEnv, DEFAULT_MAZE, UP, DOWN, LEFT, RIGHT
    from agents.q_learning import QLearningAgent

    env = MazeEnv(size=config.MAZE_SIZE)
    agent = QLearningAgent(env.state_size, env.action_space)

    # Quick re-train for the heatmap (uses saved rewards if available, else retrains)
    from train import train as run_train
    print("  Retraining Q-Learning on Maze for policy heatmap...")
    run_train("maze", "qlearning", verbose=False)

    # Read the trained Q-table by re-running (lightweight)
    # Re-train with a fresh agent to capture q_table
    agent2 = QLearningAgent(env.state_size, env.action_space)
    import config as cfg
    for _ in range(cfg.EPISODES):
        state = env.reset()
        for _ in range(cfg.MAX_STEPS_PER_EPISODE):
            action = agent2.select_action(state)
            next_state, reward, done = env.step(action)
            agent2.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent2.decay_epsilon()

    size = config.MAZE_SIZE
    action_arrows = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(size):
        for c in range(size):
            state = r * size + c
            if DEFAULT_MAZE[r, c] == 1:
                ax.add_patch(plt.Rectangle((c, size - 1 - r), 1, 1, color="#333"))
            elif (r, c) == (size - 1, size - 1):
                ax.add_patch(plt.Rectangle((c, size - 1 - r), 1, 1, color="#4CAF50", alpha=0.6))
                ax.text(c + 0.5, size - 1 - r + 0.5, "G", ha="center", va="center", fontsize=12, fontweight="bold")
            else:
                best_action = int(np.argmax(agent2.q_table[state]))
                arrow = action_arrows[best_action]
                ax.text(c + 0.5, size - 1 - r + 0.5, arrow, ha="center", va="center", fontsize=14)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True, color="gray", linewidth=0.5)
    ax.set_title("Q-Learning Learned Policy — Maze\n(arrows show greedy action per cell)", fontsize=12, fontweight="bold")
    path = "results/figures/maze_policy_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── 3. Final performance comparison bar chart ─────────────────────────────────

def plot_comparison_bar():
    results = {}
    for env_name in ENVS:
        for agent_name in AGENTS:
            df = load(env_name, agent_name)
            if df is None:
                continue
            # Average reward over last 50 episodes
            avg = df["reward"].values[-50:].mean()
            results[(env_name, agent_name)] = avg

    if not results:
        print("  No results found for bar chart.")
        return

    envs_present = [e for e in ENVS if any(k[0] == e for k in results)]
    x = np.arange(len(envs_present))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, agent_name in enumerate(AGENTS):
        vals = [results.get((env, agent_name), 0) for env in envs_present]
        bars = ax.bar(x + i * width, vals, width, label=LABELS[agent_name],
                      color=COLORS[agent_name], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([e.capitalize() for e in envs_present])
    ax.set_ylabel("Avg Reward (last 50 episodes)")
    ax.set_title("Final Performance Comparison — All Environments & Agents", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    path = "results/figures/comparison_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── 4. Reward shaping experiment — Grid env, varying obstacle penalty ─────────

def plot_reward_shaping_experiment():
    """
    Train Q-Learning on Grid with 3 different obstacle penalty values.
    Shows how reward design affects learning speed.
    """
    import config
    from environments.grid_env import GridEnv
    from agents.q_learning import QLearningAgent

    penalties = [-2, -10, -30]
    penalty_colors = ["#03A9F4", "#FF5722", "#9C27B0"]
    all_rewards = {}

    for penalty in penalties:
        rewards = []
        for _ in range(3):  # 3 seeds, average
            env = GridEnv(size=config.GRID_SIZE, obstacle_density=config.GRID_OBSTACLE_DENSITY)
            agent = QLearningAgent(env.state_size, env.action_space)

            run_rewards = []
            for _ in range(config.EPISODES):
                state = env.reset()
                total = 0.0
                for _ in range(config.MAX_STEPS_PER_EPISODE):
                    action = agent.select_action(state)
                    nr, r, done = env.step(action)
                    # Apply custom penalty
                    if r == -10:
                        r = penalty
                    agent.update(state, action, r, nr, done)
                    state = nr
                    total += r
                    if done:
                        break
                agent.decay_epsilon()
                run_rewards.append(total)
            rewards.append(run_rewards)

        all_rewards[penalty] = np.mean(rewards, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for penalty, color in zip(penalties, penalty_colors):
        smoothed = smooth(all_rewards[penalty])
        ax.plot(range(1, config.EPISODES + 1), smoothed, color=color,
                label=f"Obstacle penalty = {penalty}", linewidth=2)

    ax.set_title("Reward Shaping Experiment — Grid Environment\n(Q-Learning, varying obstacle penalty)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total Reward (smoothed, window={WINDOW})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = "results/figures/reward_shaping_grid.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    print("\nGenerating plots...\n")
    print("1. Learning curves")
    plot_learning_curves()
    print("\n2. Maze policy heatmap")
    plot_maze_policy()
    print("\n3. Comparison bar chart")
    plot_comparison_bar()
    print("\n4. Reward shaping experiment")
    plot_reward_shaping_experiment()
    print("\nAll plots saved to results/figures/\n")

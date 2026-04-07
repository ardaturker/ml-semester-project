"""
play.py — Interactive terminal script for the ML RL project.

Modes:
    human  — You play using WASD / arrow keys (Maze/Grid) or 1-4 keys (Feudal)
    watch  — A trained agent plays automatically

Usage:
    python play.py --env maze --mode human
    python play.py --env grid --mode human
    python play.py --env feudal --mode human
    python play.py --env maze --mode watch --agent qlearning
    python play.py --env feudal --mode watch --agent dqn --delay 0.6
"""

import sys
import os
import curses
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config
from environments.maze_env import MazeEnv
from environments.grid_env import GridEnv
from environments.feudal_env import FeudalEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent


# ── Factories ─────────────────────────────────────────────────────────────────

def build_env(name: str):
    if name == "maze":
        return MazeEnv(size=config.MAZE_SIZE)
    elif name == "grid":
        return GridEnv(size=config.GRID_SIZE, obstacle_density=config.GRID_OBSTACLE_DENSITY)
    elif name == "feudal":
        return FeudalEnv(max_turns=config.FEUDAL_MAX_TURNS)
    raise ValueError(f"Unknown env: {name}")


def _agent_save_path(env_name: str, agent_name: str) -> str:
    ext = ".pt" if agent_name == "dqn" else ".pkl"
    return os.path.join(os.path.dirname(__file__), f"results/agents/{env_name}_{agent_name}{ext}")


def build_and_train_agent(name: str, env, env_name: str = "", episodes: int = 300):
    """Load saved agent if available, otherwise train fresh. Sets epsilon=0 (greedy)."""
    if name == "qlearning":
        agent = QLearningAgent(env.state_size, env.action_space)
    elif name == "sarsa":
        agent = SARSAAgent(env.state_size, env.action_space)
    elif name == "dqn":
        agent = DQNAgent(env.state_size, env.action_space)
    else:
        raise ValueError(f"Unknown agent: {name}")

    save_path = _agent_save_path(env_name, name) if env_name else ""

    if save_path and os.path.exists(save_path):
        agent.load(save_path)
        print(f"Loaded saved agent from {save_path}")
    else:
        print(f"No saved agent found — training {name} ({episodes} episodes)...", end="", flush=True)
        for ep in range(episodes):
            state = env.reset()
            if name == "sarsa":
                action = agent.select_action(state)
            for _ in range(config.MAX_STEPS_PER_EPISODE):
                if name == "sarsa":
                    next_state, reward, done = env.step(action)
                    next_action = agent.select_action(next_state)
                    agent.update(state, action, reward, next_state, next_action, done)
                    state, action = next_state, next_action
                else:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                if done:
                    break
            agent.decay_epsilon()
            if (ep + 1) % 50 == 0:
                print(".", end="", flush=True)
        print(" Done.\n")

    agent.epsilon = 0.0
    return agent


# ── Grid rendering helpers ────────────────────────────────────────────────────

def _is_maze(env) -> bool:
    return isinstance(env, MazeEnv)


def _is_grid(env) -> bool:
    return isinstance(env, GridEnv)


def _is_feudal(env) -> bool:
    return isinstance(env, FeudalEnv)


def _get_goal(env):
    """Return goal position tuple, handling MazeEnv vs GridEnv attribute names."""
    if hasattr(env, 'goal_pos'):
        return env.goal_pos
    return env.goal


def _render_grid(stdscr, env, step: int, total_reward: float, done: bool, mode: str):
    """Render ASCII grid for MazeEnv / GridEnv into a curses window."""
    stdscr.clear()
    size = env.size
    env_name = "MAZE" if _is_maze(env) else "GRID"
    goal = _get_goal(env)

    # Header
    status = "DONE!" if done else "running"
    stdscr.addstr(0, 0, f" {env_name} | Mode: {mode.upper()} | Step: {step:3d} | Reward: {total_reward:+7.1f} | {status}")
    stdscr.addstr(1, 0, "─" * 40)

    # Grid rows
    for r in range(size):
        row_str = ""
        for c in range(size):
            if (r, c) == env.agent_pos:
                row_str += "[A]"
            elif (r, c) == goal:
                row_str += "[G]"
            elif _is_maze(env) and env.maze[r, c] == 1:
                row_str += "███"
            elif _is_grid(env) and (r, c) in env.obstacles:
                row_str += "███"
            else:
                row_str += " . "
        stdscr.addstr(2 + r, 0, row_str)

    # Footer
    footer_row = 2 + size + 1
    stdscr.addstr(footer_row, 0, "─" * 40)
    if mode == "human":
        stdscr.addstr(footer_row + 1, 0, " WASD / arrow keys = move  |  Q = quit  |  R = reset")
    else:
        stdscr.addstr(footer_row + 1, 0, " Watching agent...  |  Q = quit  |  any key = pause")

    if done:
        result = "🎉 GOAL REACHED!" if total_reward > 0 else " Episode ended"
        stdscr.addstr(footer_row + 2, 0, f" {result}  —  Press R to restart or Q to quit")

    stdscr.refresh()


def _render_feudal(stdscr, env, step: int, total_reward: float, done: bool, mode: str):
    """Render Feudal environment into a curses window."""
    stdscr.clear()
    stdscr.addstr(0, 0, f" FEUDAL WARFARE | Mode: {mode.upper()} | Turn: {env.turn:3d} | Reward: {total_reward:+7.1f}")
    stdscr.addstr(1, 0, "─" * 50)

    # Territory bar
    total = FeudalEnv.TOTAL_TERRITORIES
    agent_t = env.agent_territories
    enemy_t = env.enemy_territories
    neutral = total - agent_t - enemy_t
    bar = "█" * agent_t + "░" * neutral + "▒" * enemy_t
    stdscr.addstr(2, 0, f" Territories: [{bar}]  ({agent_t} vs {enemy_t})")
    stdscr.addstr(3, 0, f" Resources:   {env.resources:2d}   |  Fortified: {'YES' if env.fortified else 'no '}")
    stdscr.addstr(4, 0, "─" * 50)

    if mode == "human" and not done:
        stdscr.addstr(5, 0, " Choose your action:")
        stdscr.addstr(6, 2, "[1] ATTACK   — take an enemy territory (costs 1 resource, 60% success)")
        stdscr.addstr(7, 2, "[2] FORTIFY  — reduce enemy attack chance this turn (free)")
        stdscr.addstr(8, 2, "[3] RECRUIT  — gain +2 resources")
        stdscr.addstr(9, 2, "[4] RAID     — steal a resource (50% success)")
        stdscr.addstr(10, 0, "─" * 50)
        stdscr.addstr(11, 0, " Press 1-4 to act  |  Q = quit  |  R = reset")
    elif mode == "watch":
        stdscr.addstr(5, 0, " Watching agent...  |  Q = quit")
    if done:
        result = "YOU WIN!" if agent_t >= total or enemy_t == 0 else "YOU LOST." if agent_t == 0 else "Time's up."
        stdscr.addstr(12, 0, f" Game over: {result}  —  Press R to restart or Q to quit")

    stdscr.refresh()


def render(stdscr, env, step, total_reward, done, mode):
    if _is_feudal(env):
        _render_feudal(stdscr, env, step, total_reward, done, mode)
    else:
        _render_grid(stdscr, env, step, total_reward, done, mode)


# ── Key → Action mappings ─────────────────────────────────────────────────────

GRID_KEYMAP = {
    ord('w'): 0,  # UP
    ord('s'): 1,  # DOWN
    ord('a'): 2,  # LEFT
    ord('d'): 3,  # RIGHT
    curses.KEY_UP:    0,
    curses.KEY_DOWN:  1,
    curses.KEY_LEFT:  2,
    curses.KEY_RIGHT: 3,
}

FEUDAL_KEYMAP = {
    ord('1'): 0,  # ATTACK
    ord('2'): 1,  # FORTIFY
    ord('3'): 2,  # RECRUIT
    ord('4'): 3,  # RAID
}


def get_action(key: int, env) -> int | None:
    if _is_feudal(env):
        return FEUDAL_KEYMAP.get(key)
    return GRID_KEYMAP.get(key)


# ── Human Player ──────────────────────────────────────────────────────────────

class HumanPlayer:
    def __init__(self, env, env_name: str):
        self.env = env
        self.env_name = env_name

    def run(self):
        curses.wrapper(self._loop)

    def _loop(self, stdscr):
        curses.cbreak()
        stdscr.keypad(True)
        curses.noecho()

        state = self.env.reset()
        step = 0
        total_reward = 0.0
        done = False

        render(stdscr, self.env, step, total_reward, done, "human")

        while True:
            key = stdscr.getch()

            if key == ord('q'):
                break

            if key == ord('r') or (done and key != ord('q')):
                state = self.env.reset()
                step = 0
                total_reward = 0.0
                done = False
                render(stdscr, self.env, step, total_reward, done, "human")
                continue

            if done:
                continue

            action = get_action(key, self.env)
            if action is None:
                continue

            state, reward, done = self.env.step(action)
            step += 1
            total_reward += reward
            render(stdscr, self.env, step, total_reward, done, "human")


# ── Agent Watcher ─────────────────────────────────────────────────────────────

class AgentWatcher:
    def __init__(self, env, env_name: str, agent, delay: float = 0.4):
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.delay = delay

    def run(self):
        curses.wrapper(self._loop)

    def _loop(self, stdscr):
        curses.cbreak()
        stdscr.keypad(True)
        curses.noecho()
        stdscr.nodelay(True)  # non-blocking getch so agent can auto-play

        state = self.env.reset()
        step = 0
        total_reward = 0.0
        done = False
        paused = False

        render(stdscr, self.env, step, total_reward, done, "watch")

        while True:
            key = stdscr.getch()

            if key == ord('q'):
                break

            if key == ord('r'):
                state = self.env.reset()
                step = 0
                total_reward = 0.0
                done = False
                render(stdscr, self.env, step, total_reward, done, "watch")
                continue

            if key != -1:
                paused = not paused  # any other key toggles pause

            if paused or done:
                time.sleep(0.05)
                continue

            action = self.agent.select_action(state)
            state, reward, done = self.env.step(action)
            step += 1
            total_reward += reward
            render(stdscr, self.env, step, total_reward, done, "watch")
            time.sleep(self.delay)

            if done:
                # Brief pause then restart
                time.sleep(1.5)
                state = self.env.reset()
                step = 0
                total_reward = 0.0
                done = False
                render(stdscr, self.env, step, total_reward, done, "watch")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play or watch RL environments.")
    parser.add_argument("--env",   choices=["maze", "grid", "feudal"], default="maze")
    parser.add_argument("--mode",  choices=["human", "watch"],         default="human")
    parser.add_argument("--agent", choices=["qlearning", "sarsa", "dqn"], default="qlearning",
                        help="Agent to use in watch mode")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between steps in watch mode (default: 0.4)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes for watch mode (default: 300)")
    args = parser.parse_args()

    env = build_env(args.env)

    if args.mode == "human":
        print(f"\nStarting human play: {args.env.upper()}")
        if not _is_feudal(env):
            print("Controls: WASD or arrow keys to move | Q to quit | R to reset\n")
        else:
            print("Controls: 1=ATTACK  2=FORTIFY  3=RECRUIT  4=RAID | Q to quit | R to reset\n")
        input("Press Enter to start...")
        HumanPlayer(env, args.env).run()

    else:
        agent = build_and_train_agent(args.agent, env, env_name=args.env, episodes=args.episodes)
        print(f"Watching {args.agent} play {args.env.upper()} (delay={args.delay}s)")
        print("Any key = pause/resume | Q = quit | R = reset\n")
        input("Press Enter to start watching...")
        AgentWatcher(env, args.env, agent, delay=args.delay).run()

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()

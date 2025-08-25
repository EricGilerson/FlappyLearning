from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from screeninfo import get_monitors
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

ASSET_DIR = Path(__file__).parent / "Images"
BG_IMG_PATH = ASSET_DIR / "background-day.png"
FLOOR_IMG_PATH = ASSET_DIR / "base.png"
BIRD_IMG_PATH = ASSET_DIR / "dev.png"
PIPE_IMG_PATH = ASSET_DIR / "pipe-green.png"

SCALE = 3
PIPE_HORIZONTAL_SPACING = 200 * SCALE
FRAME_SKIP = 1
pygame.init()
hidden = pygame.HIDDEN if hasattr(pygame, "HIDDEN") else 0
pygame.display.set_mode((1, 1), hidden)
pygame.display.set_caption("Flappy RL init")


def _load_scaled(path: Path) -> pygame.Surface:
    img = pygame.image.load(str(path))
    return pygame.transform.scale(img, (img.get_width() * SCALE, img.get_height() * SCALE))


_BG_IMG = _load_scaled(BG_IMG_PATH).convert()
_FLOOR_IMG = _load_scaled(FLOOR_IMG_PATH).convert()
_BIRD_IMG = _load_scaled(BIRD_IMG_PATH).convert_alpha()
_PIPE_IMG = _load_scaled(PIPE_IMG_PATH).convert_alpha()
BG_W, BG_H = _BG_IMG.get_size()
FLOOR_W, FLOOR_H = _FLOOR_IMG.get_size()

_DEMO_PATH = Path("demos")
_DEMO_PATH.mkdir(exist_ok=True)

def save_demos(memory: PERBuffer, fname="demo.pkl"):
    with open(_DEMO_PATH / fname, "wb") as f:
        pickle.dump(memory.memory, f)

def load_demos(fname="demo.pkl"):
    fpath = _DEMO_PATH / fname
    if not fpath.exists():
        return []
    with open(fpath, "rb") as f:
        return pickle.load(f)
try:
    from GameInit import get_next_pipes_info, FlappyBirdEnv
except (ImportError, ModuleNotFoundError):
    def get_next_pipes_info(bird_rect, pipe_list):
        """Get info about the next pipe in the game."""
        if not pipe_list:
            return [(None, None), (None, None)]

        next_pipe = None
        for p in pipe_list:
            if p[0].right > bird_rect.left:
                next_pipe = p
                break

        if next_pipe:
            pipe_top, pipe_bottom = next_pipe
            horizontal_dist = pipe_top.centerx - bird_rect.centerx
            vertical_dist = (pipe_top.bottom + pipe_bottom.top) / 2 - bird_rect.centery
            return [(horizontal_dist, vertical_dist), (None, None)]
        return [(None, None), (None, None)]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DQN and buffer
# ──────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # ── branch that processes bird-specific state ───────────────
        self.bird_path = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        # ── branch that processes pipe-relative state ───────────────
        self.pipe_path = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        # ── joint layers after concatenation ────────────────────────
        self.fc_combine = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),  # ← swap in LayerNorm
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bird_feats = self.bird_path(x[:, :2])  # first two inputs
        pipe_feats = self.pipe_path(x[:, 2:])  # last two inputs
        combined = torch.cat([bird_feats, pipe_feats], dim=1)
        return self.fc_combine(combined)


class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_inc=0.0005):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.max_priority = 1.0

    def add_demos(self, demos, demo_priority_scale=2.0):
        for experience in demos:
            self.push(*experience, demo_priority=self.max_priority * demo_priority_scale)

    def push(self, state, action, reward, next_state, done, demo_priority=None):
        experience = (state, action, reward, next_state, done)

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        # Use provided priority for demos, or default max
        priority = demo_priority if demo_priority is not None else self.max_priority
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [], [], []

        probabilities = self.priorities[:len(self.memory)] ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_inc)

        return samples, indices, weights / weights.max()

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.memory)


@dataclass
class Hyperparameters:
    # learning
    gamma: float = 0.99
    lr: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 1000

    batch_size: int = 128
    min_replay_size: int = 500
    target_update: int = 100



class Agent:
    def __init__(self, state_size=4, action_size=2, hp: Hyperparameters = None, device=None):
        self.hp = hp or Hyperparameters()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = self.hp.epsilon_start
        self.memory = PERBuffer()
        self.steps = 0
        self.episodes_done = 0

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hp.lr, weight_decay=0.01)

    # ─────────────────────────────────────────────────────────────
    def decay_epsilon(self):
        frac = min(1.0, self.episodes_done / self.hp.epsilon_decay_episodes)
        self.epsilon = (
            self.hp.epsilon_start
            - frac * (self.hp.epsilon_start - self.hp.epsilon_end)
        )


    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = (
                torch.as_tensor(state, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.hp.min_replay_size:
            return

        self.steps += 1
        samples, indices, weights = self.memory.sample(self.hp.batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)
        state_batch = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weight_batch = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_vals = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        target_q_vals = reward_batch + (1 - done_batch) * self.hp.gamma * next_q_vals

        td_errors = torch.abs(current_q_values.squeeze(1) - target_q_vals).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        mse_per_sample = F.mse_loss(
            current_q_values.squeeze(1), target_q_vals, reduction="none"
        )
        weighted_loss = (mse_per_sample * weight_batch).mean()


        self.optimizer.zero_grad()
        weighted_loss.backward()
        for p in self.policy_net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps % self.hp.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, str(path))

    def load(self, path: Path):
        checkpoint = torch.load(str(path), map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.0)
        self.steps = checkpoint.get('steps', 0)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Env helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_pipes(pipes, pipe_img, floor_y, gap, w):
    right = (w if not pipes else max(pipes, key=lambda p: p[0].centerx)[0].centerx)
    if right < w + PIPE_HORIZONTAL_SPACING:
        cy = random.randint(gap // 2 + 50 * SCALE, floor_y - gap // 2 - 50 * SCALE)
        top = pipe_img.get_rect(midbottom=(right + PIPE_HORIZONTAL_SPACING, cy - gap // 2))
        bot = pipe_img.get_rect(midtop=(right + PIPE_HORIZONTAL_SPACING, cy + gap // 2))
        pipes.append((top, bot))


def _state(bird, vel, pipes, w, h, floor_y):
    dist_top = bird.top / h

    dist_ground = (floor_y - bird.bottom) / h

    dx, dy = get_next_pipes_info(bird, pipes)[0]
    if dx is None:
        pipe_x, pipe_y = 1.0, 0.0
    else:
        pipe_x, pipe_y = dx / w, dy / h

    return np.array([dist_top, dist_ground, pipe_x, pipe_y], dtype=np.float32)


def _reward(bird, prev, act, pipes, scored, died, w, h, floor_y):

    # Base reward
    r = 0.1  # Small positive reward for staying alive

    # Big reward for scoring
    if scored:
        r += 15.0

    # Find the next pipe
    next_pipe = None
    for p in pipes:
        if p[0].right > bird.left:
            next_pipe = p
            break

    if next_pipe:
        # Get pipe gap center
        gap_center = (next_pipe[0].bottom + next_pipe[1].top) / 2

        vertical_distance = abs(bird.centery - gap_center) / (h / 2)

        # Better alignment reward
        alignment_reward = 1.0 - vertical_distance
        r += alignment_reward * 0.5

        # Horizontal progress toward pipe
        if bird.right < next_pipe[0].left:
            horizontal_progress = 1.0 - ((next_pipe[0].left - bird.right) / (PIPE_HORIZONTAL_SPACING))
            r += horizontal_progress * 0.03

        # Successfully passing a pipe (additional to score reward)
        if bird.left > next_pipe[0].right and next_pipe[0].right > 0:
            r += 5.0

    # Moderate penalty for flapping (to encourage efficient flight)
    if act == 1:
        r -= 0.05

    # Penalty for getting too close to boundaries
    ceiling_danger = max(0, 1.0 - (bird.top / (h * 0.2)))
    floor_danger = max(0, 1.0 - ((floor_y - bird.bottom) / (h * 0.2)))
    r -= (ceiling_danger + floor_danger) * 0.5

    # Substantial penalty for dying
    if died:
        r -= 15.0

    return r


def train_flappy_bird(
        episodes: int = 2000,
        render: bool = True,
        log_dir: str = "logs",
        model_dir: str = "models",
        demo_path: str | None = "demos/demo.pkl",
        eval_freq: int = 50,
        save_freq: int = 100,
) -> Agent:
    log_dir = Path(log_dir);
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir);
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlappyBirdEnv(render=render, frame_skip=FRAME_SKIP)
    agent = Agent(device=device)
    best = 0
    if demo_path and Path(demo_path).exists():
        with open(demo_path, "rb") as f:
            demo_memory = pickle.load(f)
        random.shuffle(demo_memory)
        for s, a, r, s2, d in demo_memory:
            agent.memory.push(s, a, r, s2, d)
        agent.epsilon = 0.5
        print(f"Loaded {len(demo_memory)} demo transitions.")




    best_score = 0
    scores_window = deque(maxlen=100)

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        score = env.score
        scores_window.append(score)
        avg_score = np.mean(scores_window)

        if avg_score >= 10 and ep >= 500:
            print(f"Reached target performance with avg score: {avg_score:.1f}")
            agent.save(model_dir / "flappy_early_stop.pt")
            break

        if avg_score > best_score:
            best_score = avg_score
            agent.save(model_dir / "flappy_best_avg.pt")

        if ep % save_freq == 0:
            agent.save(model_dir / f"flappy_ep{ep}.pt")

        if ep % eval_freq == 0:
            eval_score = evaluate_agent_during_training(agent, render=False)
            writer.add_scalar("eval_score", eval_score, ep)

        writer.add_scalar("score", score, ep)
        writer.add_scalar("avg_score", avg_score, ep)
        writer.add_scalar("reward", total_reward, ep)
        writer.add_scalar("epsilon", agent.epsilon, ep)

        print(
            f"Ep {ep + 1}/{episodes} | "
            f"Score={score:3d} | "
            f"Avg={avg_score:.1f} | "
            f"TotalR={total_reward:6.2f} | "
            f"Eps={agent.epsilon:5.3f}"
        )

        agent.episodes_done += 1
        agent.decay_epsilon()

    agent.save(model_dir / "flappy_final.pt")
    writer.close()
    return agent


def evaluate_agent_during_training(agent, num_episodes=5, render=False):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.01

    env = FlappyBirdEnv(render=render, frame_skip=FRAME_SKIP)
    scores = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, _, done = env.step(action)
            state = next_state

        scores.append(env.score)

    agent.epsilon = original_epsilon
    return np.mean(scores)


def evaluate_agent(model_path: Path, num_episodes: int = 10, render: bool = True):
    from train import Agent
    from GameInit import FlappyBirdEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(device=device)
    agent.load(model_path)

    scores = []
    frame_skip = 1

    for i in range(num_episodes):
        env = FlappyBirdEnv(render=render, frame_skip=frame_skip)

        if render:
            pygame.display.set_caption(f"Flappy Bird RL - Evaluation {i + 1}/{num_episodes}")

        state = env.reset()
        episode_score = 0
        done = False

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()

            action = agent.act(state)

            next_state, reward, done = env.step(action)

            state = next_state

            episode_score = env.score

            if render:
                time.sleep(0.01)

        env.close()

        scores.append(episode_score)
        print(f"Eval {i + 1}/{num_episodes}: Score={episode_score}")

    avg_score = sum(scores) / len(scores)
    print(f"Average Score: {avg_score:.1f}")

    return scores

def play_flappy_bird(record: bool = True, demo_name: str = "demo.pkl") -> None:
    import sys, pickle, pygame, time
    from pathlib import Path
    from GameInit import FlappyBirdEnv

    env           = FlappyBirdEnv(render=True, frame_skip=FRAME_SKIP)
    recorder      = Agent() if record else None
    demo_folder   = Path("demos")
    demo_folder.mkdir(exist_ok=True)


    state        = env.reset()
    last_action  = 0           # 0 = no flap, 1 = flap
    done         = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    last_action = 1
                elif event.key == pygame.K_ESCAPE:
                    env.close()
                    sys.exit()

        next_state, reward, done = env.step(last_action)

        if record:
            recorder.remember(state, last_action, reward, next_state, done)

        state        = next_state
        last_action  = 0                       # auto-reset to “no flap”

    print(f"Game over! Final score: {env.score}")
    time.sleep(1.5)
    env.close()

    if record:
        with open(demo_folder / demo_name, "wb") as f:
            pickle.dump(recorder.memory.memory, f)
        print(f"Saved {len(recorder.memory)} transitions → {demo_folder/demo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flappy Bird with PyTorch reinforcement learning')
    subparsers = parser.add_subparsers(dest="command")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play Flappy Bird manually")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    train_parser.add_argument("--render", action="store_true", help="Render the game during training")
    train_parser.add_argument("--log-dir", type=str, default="logs", help="Directory for tensorboard logs")
    train_parser.add_argument("--model-dir", type=str, default="models", help="Directory for saving models")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("model_path", type=str, help="Path to the trained model")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render the game during evaluation")

    args = parser.parse_args()

    if args.command == "play":
        play_flappy_bird()
    elif args.command == "train":
        train_flappy_bird(
            episodes=args.episodes,
            render=args.render,
            log_dir=args.log_dir,
            model_dir=args.model_dir
        )
    elif args.command == "eval":
        evaluate_agent(
            model_path=Path(args.model_path),
            num_episodes=args.episodes,
            render=args.render
        )
    else:
        parser.print_help()
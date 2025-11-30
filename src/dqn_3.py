import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from representation import FeedbackConstruction
from warehouse_environment import WarehouseEnv


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------- Replay Buffer ----------------------- #

class ReplayBuffer:
    """Simple FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ----------------------- Q-Network ----------------------- #

class DQN(nn.Module):
    """Two-layer MLP for Q-value approximation."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------- Feature Encoder ----------------------- #

class FeatureEncoder:
    """Wrap FeedbackConstruction to output dense vectors compatible with the DQN."""

    def __init__(self, feedback: FeedbackConstruction) -> None:
        self.feedback = feedback
        self.use_tiles = feedback.use_tiles
        self.size = feedback.observation_size

    def encode(self, obs: np.ndarray) -> np.ndarray:
        raw = self.feedback.process_observation(obs)
        if not self.use_tiles:
            return raw.astype(np.float32)

        vec = np.zeros(self.size, dtype=np.float32)
        for idx in raw:
            if 0 <= idx < self.size:
                vec[int(idx)] = 1.0
        return vec


# ----------------------- Config ----------------------- #

@dataclass
class TrainConfig:
    # curriculum: primero entorno medio (objetos fijos), luego entorno duro (objetos aleatorios)
    num_episodes_medium: int = 4000
    num_episodes_hard: int = 6000

    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 3e-4
    buffer_capacity: int = 150_000
    min_buffer: int = 4_000
    target_update: int = 1_000       # pasos
    epsilon_start: float = 1.0
    epsilon_final: float = 0.10
    epsilon_decay_steps: int = 400_000
    max_grad_norm: float = 10.0
    max_steps_per_episode: int = 400
    seed: int = 7
    device: str = "cpu"
    # reward shaping y refuerzos
    use_shaping: bool = True
    shaping_coeff: float = 0.8       # peso de phi(s') - phi(s)
    pickup_bonus: float = 2.0        # extra por recoger
    success_bonus: float = 10.0      # extra por entregar correctamente


# ----------------------- Agent ----------------------- #

class DQNAgent:
    """DQN con Double-Q target, replay buffer, shaping y curriculum medio->dificil."""

    def __init__(self, env: WarehouseEnv, encoder: FeatureEncoder, config: TrainConfig) -> None:
        self.env = env
        self.encoder = encoder
        self.config = config
        self.device = torch.device(config.device)

        self.num_actions = env.action_space.n
        self.policy_net = DQN(self.encoder.size, self.num_actions).to(self.device)
        self.target_net = DQN(self.encoder.size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(config.buffer_capacity)
        self.total_steps = 0

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[float] = []
        self.epsilon_history: List[float] = []

    # --------- Epsilon --------- #

    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_final + (self.config.epsilon_start - self.config.epsilon_final) * (1.0 - frac)

    def act(self, state_vec: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    # --------- Potencial / shaping --------- #

    def _potential(self, obs: np.ndarray) -> float:
        """Phi(s): distancia negativa a objetivo segun estado."""
        agent_x, agent_y = float(obs[0]), float(obs[1])
        agent_pos = (agent_x, agent_y)
        has_object = obs[8] > 0.5

        if has_object:
            return -self.env._distance_to_area(agent_pos)

        objs = [
            (float(obs[2]), float(obs[3])),
            (float(obs[4]), float(obs[5])),
            (float(obs[6]), float(obs[7])),
        ]
        dists = []
        for ox, oy in objs:
            if abs(ox - agent_x) < 1e-6 and abs(oy - agent_y) < 1e-6:
                continue
            dists.append(np.hypot(ox - agent_x, oy - agent_y))

        return -min(dists) if dists else 0.0

    # --------- Exito --------- #

    @staticmethod
    def _is_success(terminated: bool, next_obs: np.ndarray, info: Any) -> bool:
        """
        Exito = entrega correcta (delivery=True en obs[10]) cuando el episodio termina.
        """
        if not terminated:
            return False

        if isinstance(info, dict) and "success" in info:
            return bool(info["success"])

        return float(next_obs[10]) > 0.5

    # --------- Optimizacion --------- #

    def _optimize(self) -> None:
        if len(self.replay) < max(self.config.min_buffer, self.config.batch_size):
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.policy_net(next_states_t)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + (~dones_t).float() * (self.config.gamma * next_q_target)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.total_steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # --------- Entrenamiento (una fase) --------- #

    def train(self, num_episodes: int, phase_name: str = "") -> None:
        success_window = 500
        recent_successes: deque = deque(maxlen=success_window)
        progress = trange(num_episodes, desc=f"Training {phase_name}", leave=True)

        for _ in progress:
            obs, _ = self.env.reset()
            state_vec = self.encoder.encode(obs)
            episode_return = 0.0
            success = False

            phi_s = self._potential(obs) if self.config.use_shaping else 0.0
            prev_obs = obs

            for step in range(self.config.max_steps_per_episode):
                epsilon = self._epsilon()
                action = self.act(state_vec, epsilon)

                next_obs, base_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                phi_s_next = self._potential(next_obs) if self.config.use_shaping else 0.0
                shaping = self.config.shaping_coeff * (phi_s_next - phi_s) if self.config.use_shaping else 0.0

                reward = float(base_reward + shaping)

                # bonus por recoger
                has_obj_prev = prev_obs[8] > 0.5
                has_obj_now = next_obs[8] > 0.5
                if (not has_obj_prev) and has_obj_now:
                    reward += self.config.pickup_bonus

                if self._is_success(terminated, next_obs, info):
                    success = True
                    reward += self.config.success_bonus

                next_state_vec = self.encoder.encode(next_obs) if not done else np.zeros_like(state_vec)

                self.replay.push(state_vec, action, reward, next_state_vec, done)

                self.total_steps += 1
                episode_return += reward
                self.epsilon_history.append(epsilon)

                if len(self.replay) >= max(self.config.min_buffer, self.config.batch_size):
                    self._optimize()

                state_vec = next_state_vec
                phi_s = phi_s_next
                prev_obs = next_obs

                if done:
                    break

            self.episode_returns.append(episode_return)
            self.episode_lengths.append(step + 1)
            recent_successes.append(1 if success else 0)
            self.success_rate.append(100.0 * np.mean(recent_successes))

            progress.set_description(
                f"{phase_name} Eps {self._epsilon():.3f} | R {episode_return:.2f} | Succ {self.success_rate[-1]:.1f}%"
            )

    # --------- Evaluacion --------- #

    def evaluate(self, num_episodes: int = 300) -> Tuple[float, float]:
        returns: List[float] = []
        successes: List[int] = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            state_vec = self.encoder.encode(obs)
            prev_obs = obs
            done = False
            ep_return = 0.0
            success = False

            phi_s = self._potential(obs) if self.config.use_shaping else 0.0

            while not done:
                action = self.act(state_vec, epsilon=0.0)  # greedy
                next_obs, base_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                phi_s_next = self._potential(next_obs) if self.config.use_shaping else 0.0
                shaping = self.config.shaping_coeff * (phi_s_next - phi_s) if self.config.use_shaping else 0.0

                reward = float(base_reward + shaping)

                has_obj_prev = prev_obs[8] > 0.5
                has_obj_now = next_obs[8] > 0.5
                if (not has_obj_prev) and has_obj_now:
                    reward += self.config.pickup_bonus

                ep_return += reward

                if self._is_success(terminated, next_obs, info):
                    success = True
                    ep_return += self.config.success_bonus

                if not done:
                    state_vec = self.encoder.encode(next_obs)
                    phi_s = phi_s_next
                    prev_obs = next_obs

            returns.append(ep_return)
            successes.append(1 if success else 0)

        return float(np.mean(returns)), float(np.mean(successes))

    # --------- Guardado --------- #

    def save(self, path: str) -> None:
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "config": self.config.__dict__,
            "feature_size": self.encoder.size,
        }
        torch.save(payload, path)


# ----------------------- Plots ----------------------- #

def plot_metrics(agent: DQNAgent, env_tag: str = "3") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(agent.episode_returns)
    axes[0, 0].set_title("Episode Returns")

    if len(agent.episode_returns) >= 200:
        window = 200
        ma = np.convolve(agent.episode_returns, np.ones(window) / window, mode="valid")
        axes[0, 1].plot(ma)
        axes[0, 1].set_title(f"Returns MA (window={window})")

    axes[1, 0].plot(agent.episode_lengths)
    axes[1, 0].set_title("Episode lengths")

    axes[1, 1].plot(agent.success_rate)
    axes[1, 1].set_title("Success rate (%)")

    plt.tight_layout()
    plt.savefig(f"plots/dqn_metrics_env_{env_tag}.png")
    plt.show()


# ----------------------- Main ----------------------- #

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainConfig(device=device)
    set_seed(config.seed)

    # Fase 1: entorno medio (objetos fijos, recoger + entregar)
    env_medium = WarehouseEnv(just_pick=False, random_objects=False)

    # Fase 2: entorno dificil (objetos aleatorios + entrega)
    env_hard = WarehouseEnv(just_pick=False, random_objects=True)

    feedback = FeedbackConstruction(
        dims=(env_medium.width, env_medium.height),
        target_area=env_medium.delivery_area,
        use_tiles=False,   # vector denso normalizado
    )
    encoder = FeatureEncoder(feedback)

    agent = DQNAgent(env_medium, encoder, config)

    # Entrenamiento en entorno medio
    agent.train(num_episodes=config.num_episodes_medium, phase_name="MEDIUM")

    # Curriculum: pasamos al entorno dificil con la misma red
    agent.env = env_hard
    agent.train(num_episodes=config.num_episodes_hard, phase_name="HARD")

    agent.env = env_hard
    avg_return, success_rate = agent.evaluate(num_episodes=300)
    print(f"[HARD ENV] Evaluation -> Avg return: {avg_return:.3f} | Success rate: {success_rate*100:.1f}%")

    agent.save(f"models/dqn_env3_medium_hard.pt")
    plot_metrics(agent, env_tag="3_hard")


if __name__ == "__main__":
    main()

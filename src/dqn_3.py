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
    """Fija semillas para reproducibilidad."""
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

class DuelingDQN(nn.Module):
    """Arquitectura dueling para estabilizar en recompensas escasas."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        values = self.value_head(feat)
        advantages = self.adv_head(feat)
        advantages = advantages - advantages.mean(dim=1, keepdim=True)
        return values + advantages


# ----------------------- Feature Encoder ----------------------- #

class FeatureEncoder:
    """Envuelve FeedbackConstruction para producir vectores densos."""

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
    # entrenamiento directo en entorno 3 (objetos aleatorios + entrega)
    num_episodes: int = 9000

    batch_size: int = 64
    gamma: float = 0.995
    lr: float = 5e-4
    buffer_capacity: int = 200_000
    min_buffer: int = 3_000
    warmup_random_steps: int = 3_000
    target_update: int = 400       # pasos
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay_steps: int = 150_000
    max_grad_norm: float = 5.0
    max_steps_per_episode: int = 320
    seed: int = 11
    device: str = "cpu"
    # reward shaping y refuerzos
    use_shaping: bool = True
    shaping_coeff: float = 1.0     # peso de phi(s') - phi(s)
    pickup_bonus: float = 5.0      # extra por recoger
    success_bonus: float = 40.0    # extra por entregar correctamente
    delivery_zone_bonus: float = 3.0  # refuerzo por llevar el objeto dentro de la zona antes de soltar


# ----------------------- Agent ----------------------- #

class DQNAgent:
    """Dueling DQN con Double-Q target, replay y shaping (sin curriculum)."""

    def __init__(self, env: WarehouseEnv, encoder: FeatureEncoder, config: TrainConfig) -> None:
        self.env = env
        self.encoder = encoder
        self.config = config
        self.device = torch.device(config.device)

        self.num_actions = env.action_space.n
        self.policy_net = DuelingDQN(self.encoder.size, self.num_actions).to(self.device)
        self.target_net = DuelingDQN(self.encoder.size, self.num_actions).to(self.device)
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
        # decay comienza despues del warmup para priorizar exploracion amplia inicial
        steps_for_decay = max(1, self.config.epsilon_decay_steps)
        frac = min(1.0, max(0, self.total_steps - self.config.warmup_random_steps) / steps_for_decay)
        return self.config.epsilon_final + (self.config.epsilon_start - self.config.epsilon_final) * (1.0 - frac)

    def act(self, state_vec: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    # --------- Heuristicas de accion --------- #

    def _force_action_if_needed(self, obs: np.ndarray, action: int) -> int:
        """
        Evita quedarse sin recoger/soltar: si ya esta en rango, fuerza Pick/Drop.
        """
        agent_pos = (float(obs[0]), float(obs[1]))
        has_object = obs[8] > 0.5

        if has_object:
            if self.env._is_in_area(agent_pos, self.env.delivery_area, margin=0.25):
                return 5  # Drop
            return action

        for i in range(3):
            ox, oy = float(obs[2 + 2 * i]), float(obs[3 + 2 * i])
            # si ya fue recogido, la obs lo pone en el agente; saltar
            if abs(ox - agent_pos[0]) < 1e-6 and abs(oy - agent_pos[1]) < 1e-6:
                continue
            dist = np.hypot(ox - agent_pos[0], oy - agent_pos[1])
            if dist <= (self.env.pickup_distance + self.env.agent_radius + 0.1):
                return 4  # Pick
        return action

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
        """Exito = entrega correcta (delivery=True en obs[10]) cuando el episodio termina."""
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

    # --------- Entrenamiento --------- #

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

                if self.total_steps < self.config.warmup_random_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state_vec, epsilon)

                action = self._force_action_if_needed(prev_obs, action)

                next_obs, base_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                phi_s_next = self._potential(next_obs) if self.config.use_shaping else 0.0
                shaping = self.config.shaping_coeff * (phi_s_next - phi_s) if self.config.use_shaping else 0.0

                reward = float(base_reward + shaping)

                has_obj_prev = prev_obs[8] > 0.5
                has_obj_now = next_obs[8] > 0.5
                if (not has_obj_prev) and has_obj_now:
                    reward += self.config.pickup_bonus

                # si ya lleva objeto y pisa la zona de entrega, refuerzo para que aprenda a dejarlo
                if has_obj_now and self.env._is_in_area(
                    (float(next_obs[0]), float(next_obs[1])),
                    self.env.delivery_area,
                    margin=0.25,
                ):
                    reward += self.config.delivery_zone_bonus

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
                action = self._force_action_if_needed(prev_obs, action)
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

    # Entrenamiento directo en entorno 3 (objetos aleatorios + entrega)
    env_hard = WarehouseEnv(just_pick=False, random_objects=True)

    feedback = FeedbackConstruction(
        dims=(env_hard.width, env_hard.height),
        target_area=env_hard.delivery_area,
        use_tiles=False,   # vector denso normalizado
    )
    encoder = FeatureEncoder(feedback)

    agent = DQNAgent(env_hard, encoder, config)

    # Entrenamiento completo en entorno aleatorio (sin curriculum)
    agent.train(num_episodes=config.num_episodes, phase_name="HARD")

    avg_return, success_rate = agent.evaluate(num_episodes=300)
    print(f"[HARD ENV] Evaluation -> Avg return: {avg_return:.3f} | Success rate: {success_rate*100:.1f}%")

    agent.save("models/dqn_env3_direct.pt")
    plot_metrics(agent, env_tag="3_direct")


if __name__ == "__main__":
    main()

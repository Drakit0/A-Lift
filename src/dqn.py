import numpy as np
from warehouse_environment import WarehouseEnv
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from collections import deque
import random
from tqdm import trange
import math
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# N-step accumulator: agrupa transiciones y genera transiciones n-step para el replay buffer
class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n = n_step
        self.gamma = gamma
        self.deque = deque()  # stores tuples (state, action, reward, next_state, done)

    def push(self, transition):
        # transition = (state, action, reward, next_state, done)
        self.deque.append(transition)

    def can_pop(self):
        return len(self.deque) >= self.n

    def pop_n_step(self):
        """
        Returns a single n-step tuple:
        (state_0, action_0, R_0^{(n)}, state_n, done_n)
        Where R_0^{(n)} = r0 + gamma*r1 + ... + gamma^{n-1} r_{n-1}
        If the trajectory ends before n steps (done), we compute until terminal and mark done.
        """
        if len(self.deque) == 0:
            return None

        # Always compute for the leftmost element (t=0)
        R = 0.0
        gamma_pow = 1.0
        state_0, action_0, _, _, _ = self.deque[0]
        state_n = None
        done_n = False

        for k, (_, _, r, next_s, done) in enumerate(self.deque):
            if k >= self.n:
                break
            R += gamma_pow * r
            gamma_pow *= self.gamma
            state_n = next_s
            done_n = done
            if done:
                # trajectory ended; consume up to and including this step
                break

        # Remove the first element from deque (we will push a n-step transition for it)
        self.deque.popleft()

        return (state_0, action_0, R, state_n, done_n)

    def flush_all(self):
        """Return remaining accumulated n-step transitions until deque is empty."""
        results = []
        while len(self.deque) > 0:
            # compute available truncated n-step for current head
            R = 0.0
            gamma_pow = 1.0
            state_0, action_0, _, _, _ = self.deque[0]
            state_n = None
            done_n = False

            for k, (_, _, r, next_s, done) in enumerate(self.deque):
                if k >= self.n:
                    break
                R += gamma_pow * r
                gamma_pow *= self.gamma
                state_n = next_s
                done_n = done
                if done:
                    break

            self.deque.popleft()
            results.append((state_0, action_0, R, state_n, done_n))

        return results

class DQNAgent:
    """DQNAgent is an implementation of the DQN
    
    Attributes:
        env: The environment in which the agent operates.
        feedback: An object that processes observations from the environment.
        learning_rate: The learning rate for updating the weights.
        discount_factor: The discount factor for future rewards.
        epsilon: The probability of choosing a random action (exploration rate).
        num_actions: The number of possible actions in the environment.
        feature_size: The size of the feature vector for each state.
        weights: The weights for each action.
        episode_returns: List of returns per episode during training.
        episode_lengths: List of episode lengths during training.
        success_rate: List of success rates during training.
        epsilon_history: History of epsilon values during training.
    """
    
    def __init__(self, env: WarehouseEnv, feedback: FeedbackConstruction, 
                 learning_rate: float = 0.5, discount_factor: float = 0.9, 
                 epsilon: float = 0.5,
                 n_step: int = 3,
                 lambda_: float = 0.0,
                 trace_online: bool = False) -> None:
        """Initializes the QLAgent with the given parameters.
        
        Args:
            env: The environment in which the agent operates.
            feedback: An object that processes observations from the environment.
            learning_rate: The learning rate for updating the weights. Defaults to 0.5.
            discount_factor: The discount factor for future rewards. Defaults to 0.9.
            epsilon: The probability of choosing a random action. Defaults to 0.5.
        """

        self.env: WarehouseEnv = env
        self.feedback: FeedbackConstruction = feedback
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.num_actions: int = env.action_space.n
        self.feature_size: int = feedback.observation_size

        self.dqn_target: DQN = DQN(self.feature_size, self.num_actions)
        self.dqn_actual: DQN = DQN(self.feature_size, self.num_actions)
        
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[float] = []
        self.epsilon_history: List[float] = []

        self.n_step: int = max(1, n_step)
        self.lambda_: float = float(lambda_)
        self.trace_online: bool = bool(trace_online)
        
        # if using eligibility traces in online mode, prepare traces per parameter
        self._eligibility_traces = None
        if self.trace_online and self.lambda_ > 0.0:
            # initialize when training starts, shapes depend on network parameters
            self._eligibility_traces = None


    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Selects an action based on the epsilon-greedy policy.
        
        Args:
            state: The current state of the environment.
            epsilon: The probability of selecting a random action. 
                If None, the default epsilon value is used.
                
        Returns:
            The selected action.
        """
        
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Random action
        
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def get_q_values(self, state: np.ndarray, actual_network: bool = True) -> np.ndarray:
        """Computes the Q-values of all actions for a given state.

        Args:
            state: The current state for which Q-values need to be computed.

        Returns:
            A numpy array of Q-values for each action in the given state.
        """
        
        features: np.ndarray = self.feedback.process_observation(state)  # returns np.array shape (feature_size,)
        input_tensor = torch.tensor(features, dtype=torch.float32)
        if actual_network:
            q_values = self.dqn_actual(input_tensor).detach().numpy()
        else:
            q_values = self.dqn_target(input_tensor).detach().numpy()
        return q_values

    def _init_parameter_traces(self):
        """Initialize eligibility traces tensors matching each trainable param."""
        self._eligibility_traces = []
        for p in self.dqn_actual.parameters():
            self._eligibility_traces.append(torch.zeros_like(p.data))

    def train(self, num_episodes: int, decay_start: float, decay_rate: float, 
              min_epsilon: float, batch_size: int = 4, c: int = 4*4,
              n_step: Optional[int] = None, lambda_: Optional[float] = None) -> None:
        """Train the DQN agent.
        If self.trace_online == True, training uses online updates with eligibility traces (no replay).
        Otherwise uses experience replay with n-step transitions (n_step controlled by self.n_step).
        """
        # allow overriding via args
        if n_step is not None:
            self.n_step = max(1, n_step)
        if lambda_ is not None:
            self.lambda_ = float(lambda_)

        success_window: int = 500  # Track success over last N episodes
        recent_successes: List[int] = []

        replay_buffer: ReplayBuffer = ReplayBuffer(1_000_000)
        n_acc = NStepAccumulator(self.n_step, self.discount_factor)

        # The optimizer and loss function (only used for replay case)
        optimizer = torch.optim.Adam(self.dqn_actual.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Initialize the target network
        self.dqn_target.load_state_dict(self.dqn_actual.state_dict())
        global_step = 0

        # if using eligibility traces online: prepare traces
        if self.trace_online and self.lambda_ > 0.0:
            self._init_parameter_traces()

        progress_bar = trange(num_episodes)

        for episode in progress_bar:
            state, _ = self.env.reset()

            # epsilon decay
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = float(np.max([min_epsilon,self.epsilon]))
            self.epsilon_history.append(self.epsilon)

            n_steps: int = 0
            total_undiscounted_return: float = 0.0

            # clear n-step accumulator at episode start
            n_acc = NStepAccumulator(self.n_step, self.discount_factor)

            while True:
                action: int = self.get_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_undiscounted_return += reward

                # Push single-step into accumulator
                n_acc.push((state, action, reward, next_state, done))

                # If enough steps accumulated, create n-step transition and push to main replay buffer
                if not self.trace_online:
                    if n_acc.can_pop():
                        nstep_trans = n_acc.pop_n_step()  # (s, a, R^{(n)}, s_n, done_n)
                        if nstep_trans is not None:
                            replay_buffer.push(nstep_trans)
                    if done:
                        # flush remaining truncated accumulations
                        remaining = n_acc.flush_all()
                        for tr in remaining:
                            replay_buffer.push(tr)

                    # Vectorized minibatch update from replay (n-step transitions)
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size)
                        states = [b[0] for b in batch]
                        actions = [b[1] for b in batch]
                        rewards = [b[2] for b in batch]           # already n-step return
                        next_states = [b[3] for b in batch]       # s_{t+n}
                        dones = [b[4] for b in batch]

                        # Process features
                        s_feats = np.stack([self.feedback.process_observation(s) for s in states], axis=0)
                        s_next_feats = np.stack([self.feedback.process_observation(sn) if sn is not None else np.zeros(self.feature_size) for sn in next_states], axis=0)

                        s_tensor = torch.tensor(s_feats, dtype=torch.float32)
                        s_next_tensor = torch.tensor(s_next_feats, dtype=torch.float32)
                        actions_tensor = torch.tensor(actions, dtype=torch.long)
                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                        dones_tensor = torch.tensor(dones, dtype=torch.bool)

                        # Current Q(s,a; theta)
                        q_values = self.dqn_actual(s_tensor)                         # (B, num_actions)
                        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (B,)

                        # Double DQN with n-step target:
                        with torch.no_grad():
                            q_next_online = self.dqn_actual(s_next_tensor)         # (B, num_actions)
                            next_actions = torch.argmax(q_next_online, dim=1)     # (B,)
                            q_next_target = self.dqn_target(s_next_tensor)        # (B, num_actions)
                            q_next_selected = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
                            # multiply future term by gamma^n
                            gamma_n = (self.discount_factor ** self.n_step)
                            target = rewards_tensor + (~dones_tensor).float() * (gamma_n * q_next_selected)

                        loss = loss_fn(current_q, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                else:
                    # trace_online == True: use online updates with eligibility traces (backward view)
                    # For stability, we use target computed with target network (one-step target) and apply
                    # eligibility trace updates on parameters (accumulating gradients).
                    s_feat = self.feedback.process_observation(state)
                    s_next_feat = self.feedback.process_observation(next_state) if next_state is not None else np.zeros(self.feature_size)
                    s_tensor = torch.tensor(s_feat, dtype=torch.float32).unsqueeze(0)  # (1, F)
                    s_next_tensor = torch.tensor(s_next_feat, dtype=torch.float32).unsqueeze(0)

                    # get current Q and target value (max over actions using target net)
                    q_all = self.dqn_actual(s_tensor)  # (1, A)
                    q_current = q_all[0, action]
                    with torch.no_grad():
                        q_next = self.dqn_target(s_next_tensor)
                        q_next_max = torch.max(q_next, dim=1)[0][0]
                        gamma = self.discount_factor
                        target = torch.tensor(reward, dtype=torch.float32)
                        if not done:
                            target = target + gamma * q_next_max

                    td_error = (target - q_current).detach()

                    # compute gradients of Q(s,a) wrt parameters
                    grads = autograd.grad(q_current, tuple(self.dqn_actual.parameters()), retain_graph=False, create_graph=False)

                    # initialize traces if None
                    if self._eligibility_traces is None:
                        self._init_parameter_traces()

                    # update eligibility traces and parameters
                    for p, g, e in zip(self.dqn_actual.parameters(), grads, self._eligibility_traces):
                        # accumulate trace: e <- gamma * lambda * e + grad Q(s,a)
                        e.mul_(gamma * self.lambda_)
                        e.add_(g.detach())
                        # parameter update: theta <- theta + alpha * td_error * e
                        p.data.add_(self.learning_rate * td_error * e)

                    # optionally periodically sync target network
                    if global_step % c == 0:
                        self.dqn_target.load_state_dict(self.dqn_actual.state_dict())

                # Update state
                state = next_state
                n_steps += 1
                global_step += 1

                if terminated or truncated:
                    break

            # Track metrics
            self.episode_returns.append(total_undiscounted_return)
            self.episode_lengths.append(n_steps)
            success: bool = terminated and reward > 0
            recent_successes.append(1 if success else 0)
            
            if len(recent_successes) > success_window:
                recent_successes.pop(0)
            
            current_success_rate: float = np.mean(recent_successes) * 100
            self.success_rate.append(current_success_rate)

            progress_bar.set_description(f"Epsilon: {self.epsilon:.3f} | Success Rate (last {success_window}): {current_success_rate:.1f}%")

    def evaluate(self, num_episodes: int) -> float:
        """Evaluate the agent's performance over a specified number of episodes.
        
        Args:
            num_episodes: The number of episodes to run the evaluation.
            
        Returns:
            The average reward obtained over the specified number of episodes.
        """
        
        total_returns: List[float] = []
        
        for _  in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return: float = 0
            terminated: bool = False
            
            while not terminated:
                action: int = self.get_action(state, self.epsilon)  # Greedy policy
                next_state, reward, terminated, _, _ = self.env.step(action)
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
            
            total_returns.append(total_undiscounted_return)
        
        avg_return: float = np.mean(total_returns)
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        success_rate: float = np.mean(np.where(np.array(total_returns) > 0, 1, 0))
        
        return avg_return, success_rate
    
    def plot_training_metrics(self, num_episodes: int, avg_return: float, window: int = 500, env_str: str = "1") -> None:
        """Plot training progress.
        
        Args:
            num_episodes: The number of episodes trained.
            avg_return: The average return achieved during evaluation.
        """
        
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Returns over time
        axes[0, 0].plot(self.episode_returns)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].grid(True)
        
        # Moving average of returns
        if len(self.episode_returns) >= window:
            moving_avg: np.ndarray = np.convolve(self.episode_returns, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Returns (Moving Avg, window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Return')
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Success rate
        axes[1, 1].plot(self.success_rate)
        axes[1, 1].set_title('Success Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success %')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/dqn_metrics_env_{env_str}_{num_episodes}_{self.learning_rate}_{self.epsilon}_{avg_return:.2f}.png')
        plt.show()

def main():
    env_str = "1"
    match env_str:
        case "1":
            just_pick = True
            random_objects = False
            success_threshold = 0.95
        case "2":
            just_pick = False
            random_objects = False
            success_threshold = 0.90
        case "3":
            just_pick = False
            random_objects = True
            success_threshold = 0.85

    # Instantiate environment and representation
    env: WarehouseEnv = WarehouseEnv(just_pick=just_pick, random_objects=random_objects)
    warehouse_width: float = env.width
    warehouse_height: float = env.height
    target_area: Tuple[float, float, float, float] = env.delivery_area
    
    feedback: FeedbackConstruction = FeedbackConstruction((warehouse_width, warehouse_height), 
                                                          target_area=target_area, 
                                                          use_tiles=False)
    
    #Initialize the agent
    learning_rate: float = 1e-4
    discount_factor: float = 0.99 # Gamma: importance of future rewards
    epsilon: float = 0.5
    
    agent: DQNAgent = DQNAgent(env,
                       feedback, 
                       learning_rate, 
                       discount_factor, 
                       epsilon)
    
    # Train the agent
    decay_start: float = 0.5 # Start epsilon decay at n% of total episodes
    decay_end: float = 0.7 # End epsilon decay at n% of total episodes (min_epsilon will be reached here)
    min_epsilon: float = 0.001
    decay_rate: float = math.exp(math.log(min_epsilon)/(math.log(decay_end - decay_start)))/100 # Control of the (exponential) decrease of epsilon
    num_episodes: int = 50_000
    batch_size: int = 64
    c: int = 2_000

    print(f"Decay rate: {decay_rate}")
    
    agent.train(num_episodes, decay_start, decay_rate, min_epsilon, batch_size, c)
    
    # Evaluate the agent
    avg_return: float
    success_rate: float
    avg_return, success_rate = agent.evaluate(num_episodes=500)

    print(f"Avg_return: {avg_return}\nSucess rate: {success_rate}\nSuccess needed: {success_threshold}")
    
    # Save the agent object into memory    
    with open(f'models/dqn_trained_env_{env_str}_{num_episodes}_{learning_rate}_{epsilon}_{avg_return:.2f}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    # Plot the training results
    agent.plot_training_metrics(num_episodes, avg_return, env_str=env_str)


if __name__ == "__main__":
    main()
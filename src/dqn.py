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
                 epsilon: float = 0.5) -> None:
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
                
    def train(self, num_episodes: int, decay_start: float, decay_rate: float, 
              min_epsilon: float, batch_size: int = 4, c: int = 4*4) -> None:
        """Train the DQN agent.
        
        Args:
            num_episodes: The number of episodes to train the agent.
            decay_start: The fraction of episodes after which epsilon decay starts.
            decay_rate: The exponential decay rate for epsilon.
            min_epsilon: The minimum value for epsilon.
        """
        
        success_window: int = 500  # Track success over last N episodes
        recent_successes: List[int] = []

        replay_buffer: ReplayBuffer = ReplayBuffer(10_000)

        # The optimizer and loss function
        optimizer = torch.optim.Adam(self.dqn_actual.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Initialize the target network with the same weights as the Q-network
        self.dqn_target.load_state_dict(self.dqn_actual.state_dict())
        global_step = 0

        progress_bar = trange(num_episodes)

        for episode in progress_bar:
            # Episode setup
            state, _ = self.env.reset()
            
            # Exponential decrease of epsilon to minimum value from marked start
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = float(np.max([min_epsilon,self.epsilon]))
            
            self.epsilon_history.append(self.epsilon)
            
            # First action        
            n_steps: int = 0
            
            # Episode generation
            total_undiscounted_return: float = 0
            
            while True:
                action: int = self.get_action(state, self.epsilon)    

                next_state, reward, terminated, truncated, _ = self.env.step(action)  
                total_undiscounted_return += reward

                # Store experience in raplay buffer
                replay_buffer.push([state, action, reward, next_state, terminated or truncated])
                    
                # Vectorized minibatch update
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    states = [b[0] for b in batch]
                    actions = [b[1] for b in batch]
                    rewards = [b[2] for b in batch]
                    next_states = [b[3] for b in batch]
                    dones = [b[4] for b in batch]

                    # Process observations to normalized vectors and stack into tensors
                    s_feats = np.stack([self.feedback.process_observation(s) for s in states], axis=0)
                    s_next_feats = np.stack([self.feedback.process_observation(sn) for sn in next_states], axis=0)

                    s_tensor = torch.tensor(s_feats, dtype=torch.float32)
                    s_next_tensor = torch.tensor(s_next_feats, dtype=torch.float32)
                    actions_tensor = torch.tensor(actions, dtype=torch.long)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, dtype=torch.bool)

                    # Current Q(s,a; theta)
                    q_values = self.dqn_actual(s_tensor)                         # (B, num_actions)
                    current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (B,)

                    '''
                    # Target: r + gamma * max_a' Q_target(s', a')  (0 if done)
                    with torch.no_grad():
                        q_next = self.dqn_target(s_next_tensor)                  # (B, num_actions)
                        q_next_max, _ = torch.max(q_next, dim=1)                # (B,)
                        target = rewards_tensor + (~dones_tensor).float() * (self.discount_factor * q_next_max)
                    '''
                    # Target: r + gamma * max_a' Q_target(s', a')  (0 if done)
                    with torch.no_grad():
                        # Double DQN:
                        # select actions with online network, evaluate with target network
                        q_next_online = self.dqn_actual(s_next_tensor)         # (B, num_actions)
                        next_actions = torch.argmax(q_next_online, dim=1)     # (B,)
                        q_next_target = self.dqn_target(s_next_tensor)        # (B, num_actions)
                        q_next_selected = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
                        target = rewards_tensor + (~dones_tensor).float() * (self.discount_factor * q_next_selected)

                    loss = loss_fn(current_q, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                '''
                # Sample a mini-batch of experiences from the buffer
                if len(replay_buffer) >= batch_size:
                    batch: List[List] = replay_buffer.sample(batch_size)

                    # Calculate target for each transition in the batch
                    for transition in batch:
                        state, action, reward, next_state, done = transition

                        # Process observations -> tensors
                        s_feat = self.feedback.process_observation(state)
                        s_tensor = torch.tensor(s_feat, dtype=torch.float32)

                        if done:
                            target = torch.tensor(state, dtype=torch.float32)
                        else:
                            next_state_features = self.feedback.process_observation(next_state)
                            next_state_features_tensor = torch.tensor(next_state_features, dtype=torch.float32)
                            q_next_state = torch.max(self.dqn_target(next_state_features_tensor)).detach()
                            target = torch.tensor(reward, dtype=torch.float32) + self.discount_factor * q_next_state

                        # if done:
                        #     target: torch.Tensor = torch.tensor(reward)
                        # else:
                        #     # Target is: reward + gamma * max_a' Q(s', a'; θ^-)
                        #     target: torch.Tensor = reward + self.discount_factor * torch.max(self.dqn_target(torch.tensor(next_state)))
                        
                        # Compute the loss for the current transition
                        current_q_value: torch.Tensor = self.dqn_actual(s_tensor)[action]  # Q(s,a; theta)
                        
                        loss: torch.Tensor = loss_fn(current_q_value, target)
                    
                        # Backpropagate and update Q-network (θ)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                '''
                
                # Periodically update the target network every C steps
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
            
            # Track success (reached target)
            success: bool = terminated and reward > 0  # Adjust threshold
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
    
    def plot_training_metrics(self, num_episodes: int, avg_return: float, window: int = 500) -> None:
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
        plt.savefig(f'plots/dqn_metrics_{num_episodes}_{self.learning_rate}_{self.epsilon}_{avg_return:.2f}.png')
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
    warehouse_width: float = 10.0
    warehouse_height: float = 10.0

    # Design the tiles
    target_area: Tuple[float, float, float, float] = (2.5, 8, 1.0, 2.0)
    
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
    decay_start: float = 0.6 # Start epsilon decay at n% of total episodes
    decay_rate: float = 0.9999 # Control of the (exponential) decrease of epsilon
    min_epsilon: float = 0.001 
    num_episodes: int = 10_000
    batch_size: int = 32
    c: int = 8*batch_size
    
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
    agent.plot_training_metrics(num_episodes, avg_return)


if __name__ == "__main__":
    main()
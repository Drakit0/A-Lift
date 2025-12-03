import numpy as np
import random
from warehouse_environment import WarehouseEnv
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

class SarsaLambdaAgent:
    """SarsaLambdaAgent is an implementation of SARSA(λ) - on-policy TD control with eligibility traces.
    
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
                 epsilon: float = 0.5, lambda_value: float = 0.5,
                 seed: Optional[int] = None) -> None:
        """Initializes the SarsaLambdaAgent with the given parameters.
        
        Args:
            env: The environment in which the agent operates.
            feedback: An object that processes observations from the environment.
            learning_rate: The learning rate for updating the weights. Defaults to 0.5.
            discount_factor: The discount factor for future rewards. Defaults to 0.9.
            epsilon: The probability of choosing a random action. Defaults to 0.5.
            seed: Random seed for reproducibility. Defaults to None.
        """
        
        self.seed: Optional[int] = seed
        
        # Create dedicated RNG for deterministic action selection
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.env: WarehouseEnv = env
        self.feedback: FeedbackConstruction = feedback
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.num_actions: int = env.action_space.n

        # Tile or linear representations
        if feedback.use_tiles:
            self.feature_size: int = feedback.iht.size
            
        else:
            self.feature_size: int = feedback.observation_size
        
        self.lambda_value: float = lambda_value

        self.weights: np.ndarray = np.ones((self.num_actions, self.feature_size)) * 0.1 
        self.elegibility_traces: np.ndarray = np.zeros((self.num_actions, self.feature_size))
        
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
        
        if self.rng.random() < epsilon and epsilon != 0.0:
            return int(self.rng.integers(0, self.num_actions))  # Random action using agent's RNG
        
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Computes the Q-values of all actions for a given state.

        Args:
            state: The current state for which Q-values need to be computed.

        Returns:
            A numpy array of Q-values for each action in the given state.
        """
        
        features: List[int] = self.feedback.process_observation(state)
        q_values: np.ndarray = np.zeros(self.num_actions)
        
        if self.feedback.use_tiles: # Tiles
            for action in range(self.num_actions):
                for feature in features:
                    q_values[action] += self.weights[action][feature]
                    
        else: # Vectors
            for action in range(self.num_actions):
                q_values[action] = np.dot(self.weights[action], features)
                        
        return q_values
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, next_action: int, terminated: bool) -> None:
        """Update the weights for the given state-action pair using the SARSA(λ) algorithm.
        
        Args:
            state: The current state.
            action: The action taken in the current state.
            reward: The reward received after taking the action.
            next_state: The state resulting from taking the action.
            next_action: The action to be taken in the next state.
            terminated: Whether the episode has terminated.
        """
        
        qs_current: np.ndarray = self.get_q_values(state)       
        q_current: float = qs_current[action]
        
        # td_error
        if terminated:
            td_error: float = reward - q_current
            
        else:
            qs_next: np.ndarray = self.get_q_values(next_state)
            q_next: float = qs_next[next_action]            
            td_error: float = reward + self.discount_factor * q_next - q_current
            
        # Clip TD error to prevent explosion
        td_error = np.clip(td_error, -100, 100)
        
        features: List[int] = self.feedback.process_observation(state)
        
        # Decay traces
        self.elegibility_traces *= self.lambda_value * self.discount_factor

        # Update traces for the current state-action
        if self.feedback.use_tiles: # Tile update - replacing traces
            for feature in features:
                self.elegibility_traces[action][feature] = 1

        else: # Vector update - accumulating traces
            # For linear function approximation: e(s,a) += gradient of Q(s,a) = features
            self.elegibility_traces[action] += features
            
        self.weights += self.learning_rate * td_error * self.elegibility_traces
        self.weights = np.clip(self.weights, -100, 100)
                
    def train(self, num_episodes: int, decay_start: float, decay_rate: float, 
              min_epsilon: float, episodes_update: int = 1000, log_dir: Optional[str] = None) -> None:
        """Train the agent using the SARSA(λ) algorithm.
        
        Args:
            num_episodes: The number of episodes to train the agent.
            decay_start: The fraction of episodes after which epsilon decay starts.
            decay_rate: The exponential decay rate for epsilon.
            min_epsilon: The minimum value for epsilon.
            episodes_update: Frequency of progress updates.
            log_dir: Directory for TensorBoard logs. If None, creates timestamped dir.
        """
        
        # Setup TensorBoard with descriptive name
        repr_type = "tile" if self.feedback.use_tiles else "vec"
        env_type = "env1" if self.feedback.just_pick else ("env3" if self.feedback.random_objects else "env2")
        
        if log_dir is None:
            log_dir = f"runs/sarsa_lambda_{env_type}_{repr_type}_lr{self.learning_rate}_g{self.discount_factor}_l{self.lambda_value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"Run 'tensorboard --logdir={os.path.dirname(log_dir)}' to view")
        
        # Log hyperparameters
        hparams = {
            'algorithm': 'SARSA(lambda)',
            'env_type': env_type,
            'representation': repr_type,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'lambda': self.lambda_value,
            'epsilon_start': self.epsilon,
            'epsilon_min': min_epsilon,
            'decay_rate': decay_rate,
            'decay_start': decay_start,
            'num_episodes': num_episodes,
            'feature_size': self.feature_size
        }
        writer.add_text('Hyperparameters', str(hparams), 0)
        
        success_window: int = 100  # Track success over last N episodes
        recent_successes: List[int] = []

        progress_bar = trange(num_episodes)

        for episode in progress_bar:
            # Seed up for reproductibility
            if self.seed is not None:
                episode_seed = self.seed + episode
                obs, _ = self.env.reset(seed=episode_seed)
            else:
                obs, _ = self.env.reset()
            
            # Reset elegibility traces
            self.elegibility_traces = np.zeros((self.num_actions, self.feature_size))

            # Exponential decrease of epsilon to minimum value from marked start
            if episode >= num_episodes*decay_start:
                self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
            
            self.epsilon_history.append(self.epsilon)
            
            # First action
            action: int = self.get_action(obs, self.epsilon)            
            
            episode_return: float = 0.0
            episode_length: int = 0
            terminated: bool = False
            truncated: bool = False
            
            while not terminated and not truncated:
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                
                next_action = self.get_action(next_obs, self.epsilon)
                
                self.update(obs, action, reward, next_obs, next_action, terminated)
                
                episode_return += reward
                episode_length += 1
                
                obs = next_obs
                action = next_action
            
            # Track metrics
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            
            # Track success (reached target)
            success: bool = terminated and episode_return > 0
            recent_successes.append(1 if success else 0)
            
            if len(recent_successes) > success_window:
                recent_successes.pop(0)
            
            current_success_rate: float = sum(recent_successes) / len(recent_successes)
            self.success_rate.append(current_success_rate)
            
            # Log to TensorBoard - Basic metrics
            writer.add_scalar('Training/Episode_Return', episode_return, episode)
            writer.add_scalar('Training/Episode_Length', episode_length, episode)
            writer.add_scalar('Training/Success_Rate', current_success_rate * 100, episode)
            writer.add_scalar('Training/Epsilon', self.epsilon, episode)
            
            # Log moving averages every 100 episodes
            if episode > 0 and episode % 100 == 0:
                window = min(100, len(self.episode_returns))
                avg_return_100 = np.mean(self.episode_returns[-window:])
                avg_length_100 = np.mean(self.episode_lengths[-window:])
                writer.add_scalar('MovingAvg/Return_100', avg_return_100, episode)
                writer.add_scalar('MovingAvg/Length_100', avg_length_100, episode)
            
            # Update progress bar description only every 100 episodes to avoid spam
            if episode % 100 == 0:
                progress_bar.set_description(f"Eps: {self.epsilon:.3f} | SR: {current_success_rate*100:.1f}% | Steps: {int(np.mean(self.episode_lengths[-success_window:]))} | Ret: {np.mean(self.episode_returns[-success_window:]):.2f}")
        
        # Log final metrics
        writer.add_scalar('Final/Success_Rate', current_success_rate * 100, num_episodes)
        writer.add_scalar('Final/Avg_Return', np.mean(self.episode_returns[-100:]), num_episodes)
        
        # Close TensorBoard writer
        writer.close()
        print(f"Training complete. TensorBoard logs saved to: {log_dir}")

    
    def evaluate(self, num_episodes: int) -> tuple[float, float]:
        """Evaluate the agent's performance over a specified number of episodes.
        
        Args:
            num_episodes: The number of episodes to run the evaluation.
            
        Returns:
            The average reward obtained over the specified number of episodes.
        """
        
        total_returns: List[float] = []
        successes: List[int] = []
        episodes_length: List[int] = []
        
        success_window: int = 100
        progress_bar = trange(num_episodes)

        for episode in progress_bar:
            # Seed each evaluation episode deterministically (offset from training)
            if self.seed is not None:
                eval_seed = self.seed + 100000 + episode  # Offset to get different episodes than training
                state, _ = self.env.reset(seed=eval_seed)
            else:
                state, _ = self.env.reset()
            total_undiscounted_return: float = 0
            terminated: bool = False
            truncated: bool = False
            step_count: int = 0
            max_eval_steps: int = 500
            success: bool = False
            
            while not terminated and not truncated and step_count < max_eval_steps:
                action: int = self.get_action(state, 0.00)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
                step_count += 1
                
                # Check for success (positive terminal reward)
                if terminated and reward > 0:
                    success = True
            
            successes.append(1 if success else 0)
            total_returns.append(total_undiscounted_return)
            episodes_length.append(step_count)
            
            # Show evaluation metrics (not training metrics)
            eval_success_rate = 100 * np.mean(successes[-success_window:])
            eval_avg_return = np.mean(total_returns[-success_window:])
            progress_bar.set_description(f"Eval | Success Rate (last {success_window}): {eval_success_rate:.1f}% | Avg Return (last {success_window}): {eval_avg_return:.2f}")
        
        avg_return: float = float(np.mean(total_returns))
        success_rate: float = float(np.mean(np.array(successes)))
        
        return avg_return, success_rate, total_returns, successes, episodes_length
    
    def plot_training_metrics(self, num_episodes: int, avg_return: float, env_variant: str, workspace_def: str, learning_rate: float, decay_start: float, decay_end: float) -> None:
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
        axes[0, 0].axvline(x=decay_start*num_episodes, linestyle=':', color='red', linewidth=2)
        axes[0, 0].axvline(x=decay_end*num_episodes, linestyle=':', color='green', linewidth=2)
        
        # Moving average of returns
        window: int = 500
        if len(self.episode_returns) >= window:
            moving_avg: np.ndarray = np.convolve(self.episode_returns, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Returns (Moving Avg, window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Return')
            axes[0, 1].grid(True)
            axes[0, 1].axvline(x=decay_start*num_episodes, linestyle=':', color='red', linewidth=2)
            axes[0, 1].axvline(x=decay_end*num_episodes, linestyle=':', color='green', linewidth=2)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        axes[1, 0].axvline(x=decay_start*num_episodes, linestyle=':', color='red', linewidth=2)
        axes[1, 0].axvline(x=decay_end*num_episodes, linestyle=':', color='green', linewidth=2)
        
        # Success rate
        axes[1, 1].plot(self.success_rate)
        axes[1, 1].set_title('Success Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success %')
        axes[1, 1].grid(True)
        axes[1, 1].axvline(x=decay_start*num_episodes, linestyle=':', color='red', linewidth=2)
        axes[1, 1].axvline(x=decay_end*num_episodes, linestyle=':', color='green', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f'images/sarsa_lambda_metrics_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{self.epsilon:.2f}_{np.mean(self.success_rate):.2f}_{avg_return:.2f}.png')
        plt.show()

    def plot_value_heatmaps(self, num_episodes: int, env_variant: str, workspace_def: str, learning_rate: float, n_tiles_height: float, n_tiles_width: float):
        """
        Plot the value weights for each action as heatmaps.
        For tile coding: reshape weights to (n_tiles_height, n_tiles_width).
        For vectorized: reshape if possible, else plot as 1D.
        """
        num_actions = self.num_actions
        n_cols = min(3, num_actions)
        n_rows = (num_actions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten() if num_actions > 1 else [axes]

        for action in range(num_actions):
            if self.feedback.use_tiles:
                n_tilings = self.feedback.num_tilings
                weights = self.weights[action]
                tile_weights = np.zeros((n_tiles_height, n_tiles_width))
                for tiling in range(n_tilings):
                    offset = tiling * n_tiles_height * n_tiles_width
                    tile_weights += weights[offset:offset + n_tiles_height * n_tiles_width].reshape(n_tiles_height, n_tiles_width)
                tile_weights /= n_tilings
                im = axes[action].imshow(tile_weights, cmap='viridis', origin='lower')
                axes[action].set_title(f"Action {action} Weights (Tiles)")
                plt.colorbar(im, ax=axes[action])
            else:
                features = self.weights[action]
                size = int(np.sqrt(len(features)))
                if size * size == len(features):
                    im = axes[action].imshow(features.reshape(size, size), cmap='viridis', origin='lower')
                    axes[action].set_title(f"Action {action} Weights (Vector 2D)")
                    plt.colorbar(im, ax=axes[action])
                else:
                    axes[action].plot(features)
                    axes[action].set_title(f"Action {action} Weights (Vector 1D)")
                    axes[action].set_xlabel("Feature Index")
                    axes[action].set_ylabel("Weight")

        # Hide unused axes if any
        for i in range(num_actions, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'images/sarsa_lambda_heatmap_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{self.epsilon:.2f}.png')
        plt.show()
    
    def plot_evaluation_metrics(self, total_returns: List[float], successes: List[int], episodes_length: List[int], success_threshold: float, num_episodes: int, env_variant: str, workspace_def: str, learning_rate: float):
        """
        Plot violin plots for evaluation metrics: returns, successes, and episode lengths.
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Violin plot for total returns
        axes[0].violinplot(total_returns, showmeans=True)
        axes[0].set_title('Total Returns')
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Return')

        # Violin plot for successes (binary)
        axes[1].violinplot(successes, showmeans=True)
        axes[1].axhline(y=success_threshold, linestyle=':', color='red', linewidth=2)
        axes[1].set_title('Successes')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Success (1=Yes, 0=No)')

        # Violin plot for episode lengths
        axes[2].violinplot(episodes_length, showmeans=True)
        axes[2].set_title('Episode Lengths')
        axes[2].set_xlabel('Episodes')
        axes[2].set_ylabel('Steps')

        plt.tight_layout()
        plt.savefig(f'images/sarsa_lambda_evaluation_violin_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{self.epsilon:.2f}.png')
        plt.show()


if __name__ == "__main__":
    # Set seed for reproducibility
    SEED = 42
    set_seed(SEED)
    
    # Select environment variant
    env_variant:str = "1"  # Change to "2" or "3" for other variants
    workspace_def:str = "t" #tile-coding or vectorized space
    
    if env_variant == "1":
        just_pick = True
        random_objects = False
        success_threshold = 0.95
        
        # Common params
        discount_factor: float = 0.99 # Gamma: importance of future rewards
        lambda_value: float = 0.8
        min_epsilon: float = 0.02
        episodes_update: int = 100

        if workspace_def == "t":
            # Agent params
            learning_rate: float = 0.0125
            epsilon: float = 0.3
            
            # Training params
            decay_start: float = 0.5 # Start epsilon decay at n% of total episodes
            decay_end: float = 0.7
            num_episodes: int = 3000
            
        else:
            # Agent params
            learning_rate: float = 0.002  
            epsilon: float = 1.0
            
            # Training params
            decay_start: float = 0.3  
            decay_end: float = 0.7
            num_episodes: int = 5000

    elif env_variant == "2":
        just_pick = False
        random_objects = False
        success_threshold = 0.9
        
        # Common params
        discount_factor: float = 0.995 # Gamma: importance of future rewards
        lambda_value: float = 0.7
        min_epsilon: float = 0.05
        episodes_update: int = 100

        if workspace_def == "t":
            # Agent params
            learning_rate: float = 0.003
            epsilon: float = 0.5
            
            # Training params
            decay_start: float = 0.3 # Start epsilon decay at n% of total episodes
            decay_end: float = 0.7
            num_episodes: int = 15000
            
        else:
            # Agent params
            learning_rate: float = 0.0001
            epsilon: float = 0.8
            lambda_value: float = 0.9
            
            # Training params
            decay_start: float = 0.5 # Start epsilon decay at n% of total episodes
            decay_end: float = 0.7
            num_episodes: int = 50000
            
    else:
        just_pick = False
        random_objects = True
        success_threshold = 0.85
        
        # Common params
        discount_factor: float = 0.995 # Gamma: importance of future rewards
        lambda_value: float = 0.85  
        min_epsilon: float = 0.03  
        episodes_update: int = 100

        if workspace_def == "t":
            # Agent params
            learning_rate: float = 0.004
            epsilon: float = 0.7 
            
            # Training params
            decay_start: float = 0.2  # Start epsilon decay at n% of total episodes
            decay_end: float = 0.6 
            num_episodes: int = 30000 
            
        else:
            # Agent params
            learning_rate: float = 0.0001 
            epsilon: float = 1.0
            lambda_value: float = 0.9
            
            # Training params
            decay_start: float = 0.2 # Start epsilon decay at n% of total episodes
            decay_end: float = 0.7
            num_episodes: int = 100000

    # Exponential decrease of epsilon
    decay_rate: float = (min_epsilon/epsilon)**(1/max(1, num_episodes*(decay_end - decay_start)))
    
    # Instantiate environment and representation
    env: WarehouseEnv = WarehouseEnv(just_pick, random_objects) #, render_mode="human")
    warehouse_width: float = 10.0
    warehouse_height: float = 10.0

    # Design the tiles
    n_tiles_width: int = 10 # Number of tiles along W
    n_tiles_height: int = 10 # Number of tiles along H  
    n_tilings: int = 8 # Overlapping tiles
    
    target_area: Tuple[float, float, float, float] = (2.5, 9.0, 5.0, 1.0)
    
    feedback: FeedbackConstruction = FeedbackConstruction(
                                (warehouse_width, warehouse_height), 
                                (n_tiles_width, n_tiles_height), 
                                n_tilings, target_area,
                                use_tiles=workspace_def == "t",
                                just_pick=just_pick,
                                random_objects=random_objects)
    
    #Initialize agent    
    agent: SarsaLambdaAgent = SarsaLambdaAgent(env,
                       feedback, 
                       learning_rate, 
                       discount_factor, 
                       epsilon,
                       lambda_value,
                       seed=SEED)
    
    # Train agent
    agent.train(num_episodes, decay_start, decay_rate, min_epsilon, episodes_update)
    
    # Evaluate agent
    avg_return: float
    success_rate: float
    avg_return, success_rate, total_returns, successes, episodes_length = agent.evaluate(num_episodes=500)
    
    print(f"Results over {num_episodes} episodes:")
    print(f"Avg return: {avg_return}\nSucess rate:{'\33[41m' if success_rate < success_threshold else '\33[42m'} {success_rate} {'\33[0m'}\nSuccess needed: {success_threshold}")

    
    # Save the agent object into memory    
    with open(f'models/sarsa_lambda_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{epsilon}_{success_rate}_{avg_return:.2f}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    # Plot the training results
    agent.plot_training_metrics(num_episodes, avg_return, env_variant, workspace_def, learning_rate, decay_start, decay_end)
    agent.plot_value_heatmaps(num_episodes, env_variant, workspace_def, learning_rate, n_tiles_width, n_tiles_height)
    agent.plot_evaluation_metrics(total_returns, successes, episodes_length, success_threshold, num_episodes, env_variant, workspace_def, learning_rate)
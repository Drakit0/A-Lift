import numpy as np
from warehouse_environment import WarehouseEnv
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


class TDLambdaAgent:
    """TDLambdaAgent is an implementation of the TD(lambda)
    
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
                 epsilon: float = 0.5, lambda_value: float = 0.5) -> None:
        """Initializes the TDLambdaAgent with the given parameters.
        
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
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Random action
        
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
        """Update the weights for the given state-action pair using the TD(lambda) algorithm.
        
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

        # Replace traces instead of update, more stable behaviour
        if self.feedback.use_tiles: # Tile update
            for feature in features:
                self.elegibility_traces[action][feature] = 1

        else: # Vector update
            self.elegibility_traces[action] = features  # Add current features to trace
            
        self.weights += self.learning_rate * td_error * self.elegibility_traces
        self.weights = np.clip(self.weights, -100, 100)
                
    def train(self, num_episodes: int, decay_start: float, decay_rate: float, 
              min_epsilon: float, episodes_update: int = 1000) -> None:
        """Train the agent using the TD(lambda) algorithm.
        
        Args:
            num_episodes: The number of episodes to train the agent.
            decay_start: The fraction of episodes after which epsilon decay starts.
            decay_rate: The exponential decay rate for epsilon.
            min_epsilon: The minimum value for epsilon.
        """
        
        success_window: int = 100  # Track success over last N episodes
        recent_successes: List[int] = []

        for episode in range(num_episodes):
            # Episode setup
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
            
            if episode % episodes_update == 0:                      
                print(f"Episode {episode}, Total undiscounted return: {episode_return}, Epsilon: {self.epsilon}")
                # print(f"Steps: {n_steps}")
                # print(f"Success Rate (last {success_window}): {current_success_rate:.1f}%")
                # print(f"Avg Return (last {success_window}): {np.mean(self.episode_returns[-success_window:]):.2f}")
                # you can save the current state of the agent, if you find it useful    

    
    def evaluate(self, num_episodes: int) -> tuple[float, float]:
        """Evaluate the agent's performance over a specified number of episodes.
        
        Args:
            num_episodes: The number of episodes to run the evaluation.
            
        Returns:
            The average reward obtained over the specified number of episodes.
        """
        
        total_returns: List[float] = []
        successes: List[int] = []
        
        for episode  in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return: float = 0
            terminated: bool = False
            truncated: bool = False
            step_count: int = 0
            max_eval_steps: int = 500
            
            while not terminated and not truncated and step_count < max_eval_steps:
                action: int = self.get_action(state, 0.0)  # Evaluate the policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                if step_count % 50 == 0:
                    print(f"  Ep {episode}, Step {step_count}, Pos: {state[:2]}, Action: {action}")
                
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
                step_count += 1
                
            if step_count >= max_eval_steps:
                print(f"  Episode {episode} hit step limit - agent got stuck")
            
            successes.append(1 if reward > 0 else 0)
            total_returns.append(total_undiscounted_return)
        
        avg_return: float = float(np.mean(total_returns))
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        success_rate: float = float(np.mean(np.array(successes)))
        
        return avg_return, success_rate
    
    def plot_training_metrics(self, num_episodes: int, avg_return: float) -> None:
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
        window: int = 100
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
        plt.savefig(f'plots/td_lambda_metrics_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{epsilon}_{success_rate}_{avg_return:.2f}.png')
        plt.show()


if __name__ == "__main__":
    # Select environment variant
    env_variant:str = "1"  # Change to "2" or "3" for other variants
    workspace_def:str = "v" #tile-coding or vectorized space
    
    if env_variant == "1":
        just_pick = True
        random_objects = False
        success_threshold = 0.95
        
        # Common params
        discount_factor: float = 0.99 # Gamma: importance of future rewards
        lambda_value: float = 0.8
        decay_rate: float = 0.995 # Control of the (exponential) decrease of epsilon
        min_epsilon: float = 0.02
        episodes_update: int = 100

        if workspace_def == "t":
            # Agent params
            learning_rate: float = 0.0125
            epsilon: float = 0.3
            
            # Training params
            decay_start: float = 0.5 # Start epsilon decay at n% of total episodes
            num_episodes: int = 5000
            
        else:
            # Agent params
            learning_rate: float = 0.01
            epsilon: float = 0.5
            
            # Training params
            decay_start: float = 0.6 # Start epsilon decay at n% of total episodes
            num_episodes: int = 15000

    elif env_variant == "2":
        just_pick = False
        random_objects = False
        success_threshold = 0.9
        
        # Common params
        discount_factor: float = 0.995 # Gamma: importance of future rewards
        lambda_value: float = 0.7
        decay_rate: float = 0.998 # Control of the (exponential) decrease of epsilon
        min_epsilon: float = 0.05
        episodes_update: int = 100

        if workspace_def == "t":
            # Agent params
            learning_rate: float = 0.003
            epsilon: float = 0.5
            
            # Training params
            decay_start: float = 0.4 # Start epsilon decay at n% of total episodes
            num_episodes: int = 50000
            
        else:
            # Agent params
            learning_rate: float = 0.005
            epsilon: float = 0.5
            
            # Training params
            decay_start: float = 0.3 # Start epsilon decay at n% of total episodes
            num_episodes: int = 30000
            
    else:
        just_pick = False
        random_objects = True
        success_threshold = 0.85
        
        # Agent params
        learning_rate: float = 0.1
        discount_factor: float = 0.99 # Gamma: importance of future rewards
        epsilon: float = 0.5
        lambda_value: float = 0.7
        
        # Training params
        decay_start: float = 0.7 # Start epsilon decay at n% of total episodes
        decay_rate: float = 0.999 # Control of the (exponential) decrease of epsilon
        min_epsilon: float = 0.05
        num_episodes: int = 10000
        episodes_update: int = 100

    
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
                                just_pick=just_pick)
    
    #Initialize agent    
    agent: TDLambdaAgent = TDLambdaAgent(env,
                       feedback, 
                       learning_rate, 
                       discount_factor, 
                       epsilon,
                       lambda_value)
    
    # Train agent
    agent.train(num_episodes, decay_start, decay_rate, min_epsilon, episodes_update)
    
    # Evaluate agent
    avg_return: float
    success_rate: float
    avg_return, success_rate = agent.evaluate(num_episodes=500)
    
    print(f"Avg_return: {avg_return}\nSucess rate: {success_rate}\nSuccess needed: {success_threshold}")

    
    # Save the agent object into memory    
    with open(f'models/td_lambda_{env_variant}_{workspace_def}_{num_episodes}_{learning_rate}_{epsilon}_{success_rate}_{avg_return:.2f}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    # Plot the training results
    agent.plot_training_metrics(num_episodes, avg_return)

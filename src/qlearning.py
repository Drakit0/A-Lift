import numpy as np
from navigation_environment import Navigation
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


class QLAgent:
    """QLAgent is an implementation of the q-learing
    
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
    
    def __init__(self, env: Navigation, feedback: FeedbackConstruction, 
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

        self.env: Navigation = env
        self.feedback: FeedbackConstruction = feedback
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.num_actions: int = env.action_space.n
        self.feature_size: int = feedback.iht.size

        #TODO: try other params (kaiming?)
        self.weights: np.ndarray = np.zeros((self.num_actions, self.feature_size))
        
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
            return np.argmax(q_values)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Computes the Q-values of all actions for a given state.

        Args:
            state: The current state for which Q-values need to be computed.

        Returns:
            A numpy array of Q-values for each action in the given state.
        """
        
        features: List[int] = self.feedback.process_observation(state)
        q_values: np.ndarray = np.zeros(self.num_actions)
        
        # Calculate the values of each action for the given state (linear approximation)
        for action in range(self.num_actions):
            for feature in features:
                q_values[action] += self.weights[action][feature]
            
        return q_values
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, next_action: int, terminated: bool) -> None:
        """Update the weights for the given state-action pair using the q-learning algorithm.
        
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
            q_next: float = np.max(self.get_q_values(next_state))
            td_error: float = reward + self.discount_factor * q_next - q_current
        
        features: List[int] = self.feedback.process_observation(state)
        
        self.weights[action][features] += self.learning_rate * td_error

                
    def train(self, num_episodes: int, decay_start: float, decay_rate: float, 
              min_epsilon: float) -> None:
        """Train the agent using the QLearning algorithm.
        
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
            state, _ = self.env.reset()
            
            # Exponential decrease of epsilon to minimum value from marked start
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon,self.epsilon])
            
            self.epsilon_history.append(self.epsilon)
            
            # First action
            action: int = self.get_action(state, self.epsilon)            
            n_steps: int = 0
            
            # Episode generation
            total_undiscounted_return: float = 0
            
            while True:                                        
                next_state, reward, terminated, truncated, _ = self.env.step(action)  
                total_undiscounted_return += reward  
                        
                next_action = self.get_action(next_state, self.epsilon)
                self.update(state, action, reward, next_state, next_action, terminated)  
                  
                state = next_state
                action = next_action                
                n_steps += 1
                
                if terminated or truncated:
                    break
            
            # Track metrics
            self.episode_returns.append(total_undiscounted_return)
            self.episode_lengths.append(n_steps)
            
            # Track success (reached target)
            success: bool = terminated and total_undiscounted_return > -20  # Adjust threshold
            recent_successes.append(1 if success else 0)
            
            if len(recent_successes) > success_window:
                recent_successes.pop(0)
            
            current_success_rate: float = np.mean(recent_successes) * 100
            self.success_rate.append(current_success_rate)
            
            episodes_update: int = 1000 # Display updates
            
            if episode % episodes_update == 0:                      
                print(f"Episode {episode}, Total undiscounted return: {total_undiscounted_return}, Epsilon: {self.epsilon}")
                # print(f"Steps: {n_steps}")
                # print(f"Success Rate (last {success_window}): {current_success_rate:.1f}%")
                # print(f"Avg Return (last {success_window}): {np.mean(self.episode_returns[-success_window:]):.2f}")
                # you can save the current state of the agent, if you find it useful    

    
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
        
        return avg_return
    
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
        plt.savefig(f'plots/qlearning_metrics_{num_episodes}_{self.learning_rate}_{self.epsilon}_{avg_return:.2f}.png')
        plt.show()


if __name__ == "__main__":
    
    # Instantiate environment and representation
    env: Navigation = Navigation()
    warehouse_width: float = 10.0
    warehouse_height: float = 10.0

    # Design the tiles
    n_tiles_width: int = 10 # Number of tiles along W
    n_tiles_height: int = 10 # Number of tiles along H  
    n_tilings: int = 8 # Overlapping tiles
    
    target_area: Tuple[float, float, float, float] = (2.5, 8, 1.0, 2.0)
    
    feedback: FeedbackConstruction = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    #Initialize the agent
    learning_rate: float = 0.1
    discount_factor: float = 0.99 # Gamma: importance of future rewards
    epsilon: float = 0.1
    
    agent: QLAgent = QLAgent(env,
                       feedback, 
                       learning_rate, 
                       discount_factor, 
                       epsilon)
    
    # Train the agent
    decay_start: float = 0.9 # Start epsilon decay at n% of total episodes
    decay_rate: float = 0.9 # Control of the (exponential) decrease of epsilon
    min_epsilon: float = 0.05 
    num_episodes: int = 10000
    
    agent.train(num_episodes, decay_start, decay_rate, min_epsilon)
    
    # Evaluate the agent
    avg_return: float = agent.evaluate(num_episodes=1)
    
    # Save the agent object into memory    
    with open(f'models/qlearning_trained_{num_episodes}_{learning_rate}_{epsilon}_{avg_return:.2f}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    # Plot the training results
    agent.plot_training_metrics(num_episodes, avg_return)

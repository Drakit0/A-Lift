import numpy as np
from navigation_environment import Navigation
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt

class SarsaAgent:
    """
    SarsaAgent is an implementation of the SARSA(0) algorithm for reinforcement learning.
    Attributes:
        env (gym.Env): The environment in which the agent operates.
        feedback (object): An object that processes observations from the environment.
        learning_rate (float): The learning rate for updating the weights.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The probability of choosing a random action (exploration rate).
        num_actions (int): The number of possible actions in the environment.
        feature_size (int): The size of the feature vector for each state.
        weights (list of np.ndarray): The weights for each action.
    Methods:
        __init__(env, gateway, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
            Initializes the SarsaAgent with the given parameters.
        get_action(state, epsilon=None):
            Returns an action based on the epsilon-greedy policy.
        get_q_values(state):
            Computes the Q-values for all actions given the current state.
        update(state, action, reward, next_state, next_action):
            Updates the weights based on the SARSA update rule.
        train(num_episodes):
            Trains the agent for a specified number of episodes.
        evaluate(num_episodes):
            Evaluates the agent's performance over a specified number of episodes.
    """
    def __init__(self, env, feedback, learning_rate=0.5, discount_factor=0.9, epsilon=0.5):
        # It's better not to touch these lines
        self.env = env
        self.feedback = feedback
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = env.action_space.n
        self.feature_size = feedback.iht.size

        #TODO: try other params
        self.weights = np.zeros((self.num_actions, self.feature_size))
        
        self.episode_returns = []
        self.episode_lengths = []
        self.success_rate = []
        self.epsilon_history = []

    def get_action(self, state, epsilon=None):
        """
        Selects an action based on the epsilon-greedy policy.
        Parameters:
        state (object): The current state of the environment.
        epsilon (float, optional): The probability of selecting a random action. 
                                   If None, the default epsilon value is used.
        Returns:
        int: The selected action.
        """
        
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Random action
        
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def get_q_values(self, state):
        """
        Computes the Q-values of all actions for a given state.

        Parameters:
        state (object): The current state for which Q-values need to be computed.

        Returns:
        np.ndarray: A numpy array of Q-values for each action in the given state.
        """
        
        features = self.feedback.process_observation(state)
        q_values = np.zeros(self.num_actions)
        
        # Calculate the values of each action for the given state (linear approximation)
        for action in range(self.num_actions):
            for feature in features:
                q_values[action] += self.weights[action][feature]
            
        return q_values
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        """
        Update the weights for the given state-action pair using the SARSA(0) algorithm.
        Parameters:
        state (object): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (object): The state resulting from taking the action.
        next_action (int): The action to be taken in the next state.
        Returns:
        None
        """
        qs_current = self.get_q_values(state)       
        q_current = qs_current[action]
        
        # td_error
        if terminated:
            td_error = reward - q_current
            
        else:
            qs_next = self.get_q_values(next_state)
            q_next = qs_next[next_action]            
            td_error = reward + self.discount_factor * q_next - q_current
        
        features = self.feedback.process_observation(state)
        
        self.weights[action][features] += self.learning_rate * td_error

                
    def train(self, num_episodes, decay_start, decay_rate, min_epsilon):
        """
        Train the agent using the SARSA(0) algorithm.
        Parameters:
        num_episodes (int): The number of episodes to train the agent.
        The method runs the training loop for the specified number of episodes.
        In each episode, the agent interacts with the environment, selects actions
        based on the current policy, and updates the policy using the SARSA(0) update rule.
        The total reward for each episode is printed every 100 episodes.
        Returns:
        None
        """
        
        success_window = 100  # Track success over last N episodes
        recent_successes = []

        for episode in range(num_episodes):
            # Episode setup
            state, _ = self.env.reset()
            
            # Exponential decrease of epsilon to minimum value from marked start
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon,self.epsilon])
            
            self.epsilon_history.append(self.epsilon)
            
            # First action
            action = self.get_action(state, self.epsilon)            
            n_steps = 0
            
            # Episode generation
            total_undiscounted_return = 0
            
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
            success = terminated and total_undiscounted_return > -20  # Adjust threshold
            recent_successes.append(1 if success else 0)
            
            if len(recent_successes) > success_window:
                recent_successes.pop(0)
            
            current_success_rate = np.mean(recent_successes) * 100
            self.success_rate.append(current_success_rate)
            
            episodes_update = 1000 # Display updates
            
            if episode % episodes_update == 0:                      
                print(f"Episode {episode}, Total undiscounted return: {total_undiscounted_return}, Epsilon: {self.epsilon}")
                # print(f"Steps: {n_steps}")
                # print(f"Success Rate (last {success_window}): {current_success_rate:.1f}%")
                # print(f"Avg Return (last {success_window}): {np.mean(self.episode_returns[-success_window:]):.2f}")
                # you can save the current state of the agent, if you find it useful    

    
    def evaluate(self, num_episodes):
        """
        Evaluate the agent's performance over a specified number of episodes.
        Parameters:
        num_episodes (int): The number of episodes to run the evaluation.
        Returns:
        float: The average reward obtained over the specified number of episodes.
        This method runs the agent in the environment for a given number of episodes
        using a greedy policy (epsilon=0). It collects the total reward for each episode
        and computes the average reward over all episodes. The average reward is printed
        and returned.
        Note:
        - The environment is reset at the beginning of each episode.
        - The agent's action is determined by the `get_action` method with epsilon set to 0.
        """
        
        total_returns = []
        
        for _  in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return = 0
            terminated = False
            
            while not terminated:
                action = self.get_action(state, epsilon=self.epsilon)  # Greedy policy
                next_state, reward, terminated, _, _ = self.env.step(action)
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
            
            total_returns.append(total_undiscounted_return)
        
        avg_return = np.mean(total_returns)
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        
        return avg_return
    
    def plot_training_metrics(self, num_episodes, avg_return):
        """Plot training progress"""
        
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Returns over time
        axes[0, 0].plot(self.episode_returns)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].grid(True)
        
        # Moving average of returns
        window = 100
        if len(self.episode_returns) >= window:
            moving_avg = np.convolve(self.episode_returns, np.ones(window)/window, mode='valid')
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
        plt.savefig(f'plots/training_metrics_{num_episodes}_{self.learning_rate}_{self.epsilon}_{avg_return:.2f}.png')
        plt.show()


if __name__ == "__main__":
    
    # Instantiate environment, representation and agent
    env = Navigation()
    warehouse_width = 10.0
    warehouse_height = 10.0

    # Design the tiles
    n_tiles_width = 10 # Number of tiles along W
    n_tiles_height = 10 # Number of tiles along H  
    n_tilings = 8 # Overlapping tiles
    
    target_area = (2.5, 8, 1.0, 2.0)
    
    feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    #Initialize the agent
    learning_rate = 0.1
    discount_factor = 0.99 # Gamma: importance of future rewards
    epsilon = 0.1
    
    agent = SarsaAgent(env,
                       feedback, 
                       learning_rate, 
                       discount_factor, 
                       epsilon)
    
    # Train the agent
    decay_start = 0.9 # Start epsilon decay at n of total episodes
    decay_rate = 0.9 # Control of the (exponential) decrease of epsilon
    min_epsilon = 0.05 
    num_episodes = 10000
    
    agent.train(num_episodes, decay_start, decay_rate, min_epsilon)
    
    # Evaluate the agent
    avg_return = agent.evaluate(num_episodes=1)
    
    # Save the agent object into memory    
    with open(f'models/trained_a_lift_{num_episodes}_{learning_rate}_{epsilon}_{avg_return:.2f}.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    # Plot the training results
    agent.plot_training_metrics(num_episodes, avg_return)

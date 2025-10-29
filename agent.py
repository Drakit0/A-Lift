import numpy as np
from navigation_environment import Navigation
from representation import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

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
        self.feature_size = feedback.iht.size # If you're going to add more variables (features) besides tile coding, reserve space for them here.
        ##############################

        # We give you the weights initialized to zero. But this is arbitrary. You can change it if you want.
        self.weights = [np.zeros(self.feature_size) for _ in range(self.num_actions)]
        
        # You will need to use strategies to monitor the agent's learning.
        # Add here the attributes you need to do it.

        ##############################

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
        q_values = np.array([0,0,0,0])
        # Calculate the values of each action for the given state (linear 
        # approximation). Add your code here


        ###################################
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
        # Add your code here to update the agent's weights
        
        #############################################
        
    def train(self, num_episodes):
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
        # Play with these three hyperparameters
        decay_start = .9 # between 0 and 1. 
        decay_rate = .9 # control of the (exponential) decrease of epsilon
        min_epsilon = .5 # minimum value of epsilon
        ####################################
        for episode in range(num_episodes):
            # Episode setup
            state, _ = self.env.reset()
            # Exponential decrease of epsilon to minimum value from marked start
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon,self.epsilon])
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

            # Here you can also change the frequency with which you display
            # the results in the console, and even disable it.
            episodes_update = 1000
            if episode % episodes_update == 0:                      
                print(f"Episode {episode}, Total undiscounted return: {total_undiscounted_return}, Epsilon: {self.epsilon}")
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
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return = 0
            terminated = False
            
            while not terminated:
                action = self.get_action(state, epsilon=0.01)  # Greedy policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.env.render()
                state = next_state
                total_undiscounted_return += reward
            
            total_returns.append(total_undiscounted_return)
        
        avg_return = np.mean(total_returns)
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return}")
        return avg_return


if __name__ == "__main__":
    # Instantiate environment, representation and agent
    # Don't touch
    env = Navigation()
    warehouse_width = 10.0
    warehouse_height = 10.0
    ################
    # Design the tiles
    n_tiles_width = 1
    n_tiles_height = 1
    n_tilings = 1
    
    target_area = (2.5, 8, 1.0, 2.0)

    feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    agent = SarsaAgent(env, feedback, learning_rate=1, discount_factor=0.99, epsilon=0.5)
    
    # Train the agent
    agent.train(num_episodes=10000)
    
    # Save the agent object into memory    
    with open('agent_group_xx_a.pkl', 'wb') as f:
        pickle.dump(agent, f)

    # Evaluate the agent
    agent.evaluate(num_episodes=1)

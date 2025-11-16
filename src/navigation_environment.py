import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Optional, Tuple, List, Dict, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Navigation(gym.Env):
    """Navigation environment for reinforcement learning.
    
    Attributes:
        width: Width of the navigation environment.
        height: Height of the navigation environment.
        action_space: The discrete action space (Up, Down, Left, Right).
        observation_space: The continuous observation space.
        obstacles: List of obstacles defined by (x, y, width, height).
        target_area: The target area defined by (x, y, width, height).
        agent_radius: Radius of the agent.
        agent_velocity: Velocity of the agent.
        pickup_distance: Distance within which the agent can pick up objects.
        fig: Figure for rendering.
        ax: Axis for rendering.
        agent_pos: Current position of the agent.
        target: Whether the target has been reached.
        collision: Whether a collision has occurred.
        steps: Number of steps taken in current episode.
        max_steps: Maximum number of steps allowed per episode.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Initialize the Navigation environment.
        
        Args:
            render_mode: The rendering mode ('human', 'rgb_array', or None).
        """
        
        super(Navigation, self).__init__()
        self.render_mode: Optional[str] = render_mode
        
        # Define the size of the Navigation environment
        self.width: float = 10.0
        self.height: float = 10.0
        
        # Define action and observation space
        self.action_space: spaces.Discrete = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space: spaces.Box = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        
        # Define the obstacles (x, y, width, height)
        self.obstacles: List[Tuple[float, float, float, float]] = [
            (1.9, 1.0, 0.2, 5.0),
            (4.9, 1.0, 0.2, 5.0),
            (7.9, 1.0, 0.2, 5.0)
        ]
        
        # Define the target area
        self.target_area: Tuple[float, float, float, float] = (2.5, 8, 1.0, 2.0)  # x, y, width, height
        
        # Define agent properties
        self.agent_radius: float = 0.2
        self.agent_velocity: float = 0.25
        self.pickup_distance: float = 0.3
        
        # Add a variable to store the figure and axis for rendering
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

        # Initialize the state
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.
        
        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        
        # Reset agent position
        self.agent_pos: Tuple[float, float] = self._get_random_empty_position()  
            
        # Reset target flag
        self.target: bool = False

        # Reset collision flag
        self.collision: bool = False

        self.steps: int = 0
        self.max_steps: int = 20000
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Perform a step in the environment based on the given action.
        
        Args:
            action: The action to be taken by the agent.
            
        Returns:
            A tuple containing:
                - observation: The observation after taking the action.
                - reward: The reward received.
                - terminated: Whether the episode has terminated.
                - truncated: Whether the episode was truncated.
                - info: Additional information about the step.
        """
        
        self.steps += 1
        terminated: bool = False
        truncated: bool = False

        # Move the agent
        new_pos: Tuple[float, float] = self._get_new_position(action)
        
        # Check for collisions
        if not self._is_collision(new_pos):
            self.agent_pos = new_pos
            
        else:
            self.agent_pos = new_pos
            self.collision = True
            terminated = True
            
        # Check for goal
        if self._is_in_area(new_pos, self.target_area, self.agent_radius):
            self.target = True
            terminated = True

        # Reward calculation # TODO: Try other rewards
        reward: float
        
        if self.target:
            reward = 10
            
        elif self.collision:
            reward = -10
            
        else:
            # reward = -(abs(new_pos[0] - (self.target_area[0] + self.target_area[2]/2))+
            #            abs(new_pos[1] - (self.target_area[1] + self.target_area[3]/2)))
            reward = -0.1

        if self.steps >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {}

        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation.
        
        Returns:
            The current observation array.
        """
        
        obs: np.ndarray = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        obs[2] = self.collision
        obs[3] = self.target

        return obs
    
    def _get_new_position(self, action: int) -> Tuple[float, float]:
        """Calculate the new position of the agent based on the given action.

        Args:
            action: The action to be taken by the agent. 
                0 - Move Up
                1 - Move Down
                2 - Move Left
                3 - Move Right

        Returns:
            A tuple representing the new position (x, y) of the agent.
        """
        
        if action == 0:  # Up
            return (self.agent_pos[0], min(self.height - self.agent_radius, self.agent_pos[1] + self.agent_velocity))
        
        elif action == 1:  # Down
            return (self.agent_pos[0], max(self.agent_radius, self.agent_pos[1] - self.agent_velocity))
        
        elif action == 2:  # Left
            return (max(self.agent_radius, self.agent_pos[0] - self.agent_velocity), self.agent_pos[1])
        
        elif action == 3:  # Right
            return (min(self.width - self.agent_radius, self.agent_pos[0] + self.agent_velocity), self.agent_pos[1])
    
    def _is_collision(self, pos: Tuple[float, float]) -> bool:
        """Check if the given position results in a collision.
        
        Args:
            pos: A tuple representing the (x, y) coordinates of the position to check.
            
        Returns:
            True if there is a collision, False otherwise.
        """
        
        # Check for collisions with walls
        if (pos[0] <= self.agent_radius or pos[0] >= self.width - self.agent_radius or
            pos[1] <= self.agent_radius or pos[1] >= self.height - self.agent_radius):
            return True
        
        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if self._is_in_area(pos, obstacle, self.agent_radius):
                return True
        
        return False
    
    def _get_random_empty_position(self) -> Tuple[float, float]:
        """Generate a random position within the environment that does not collide with any obstacles.

        Returns:
            A tuple (x, y) representing the coordinates of a random, collision-free position.
        """
        
        while True:
            pos = (np.random.uniform(self.agent_radius, self.width - self.agent_radius),
                   np.random.uniform(self.agent_radius, self.height - self.agent_radius))
            
            if not self._is_collision(pos):
                return pos
    
    def _get_random_position_on_obstacle(self, obstacle: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Generate a random position on the given obstacle.

        Args:
            obstacle: A tuple representing the obstacle with the format (x, y, width, height).

        Returns:
            A tuple (x, y) representing the random position on the obstacle.
        """
        
        aux: float = np.random.uniform(0,1)
        x: float
        
        if aux < 0.5:
            x = obstacle[0] + 0.25 * obstacle[2]
            
        else:
            x = obstacle[0] + 0.75 * obstacle[2]
            
        alpha: float = 0.5
        y: float = np.random.uniform(obstacle[1] + alpha, obstacle[1] + obstacle[3] - alpha)
        
        return (x, y)
    
    @staticmethod
    def _distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points.

        Args:
            pos1: A tuple representing the (x, y) coordinates of the first point.
            pos2: A tuple representing the (x, y) coordinates of the second point.

        Returns:
            The Euclidean distance between the two points.
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def _is_in_area(pos: Tuple[float, float], area: Tuple[float, float, float, float], 
                     margin: float = 0) -> bool:
        """Check if a position is within a specified rectangular area with an optional margin.

        Args:
            pos: A tuple (x, y) representing the position to check.
            area: A tuple (x, y, width, height) representing the rectangular area.
            margin: An optional margin to consider around the area. Defaults to 0.

        Returns:
            True if the position is within the area (including the margin), False otherwise.
        """
        return (area[0] - margin <= pos[0] <= area[0] + area[2] + margin and
                area[1] - margin <= pos[1] <= area[1] + area[3] + margin)

    def render(self, mode: str = 'human', debug: bool = False) -> Optional[np.ndarray]:
        """Renders the current state of the navigation environment.
        
        Args:
            mode: The mode in which to render the environment. Default is 'human'.
                If 'rgb_array', the function returns an RGB array of the rendered image.
            debug: If True, additional debug information is rendered. Default is False.
            
        Returns:
            If mode is 'rgb_array', returns an RGB array of the rendered image.
            Otherwise, returns None.
        """
        
        if self.render_mode is None:
            return  # no render
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')

        # Draw obstacles
        for obstacle in self.obstacles:
            self.ax.add_patch(Rectangle(obstacle[:2], obstacle[2], obstacle[3], fill=False, edgecolor='brown'))

        # Draw target area
        self.ax.add_patch(Rectangle(self.target_area[:2], self.target_area[2], self.target_area[3], 
                                    fill=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))

        # Draw agent
        agent_color: str = 'orange' 
        self.ax.add_patch(Circle(self.agent_pos, radius=self.agent_radius, fill=True, facecolor=agent_color))

        if debug:
            self._render_debug_tiles()

        plt.title('Navigation Environment')
        plt.draw()
        plt.pause(0.1)

        # Save the figure
        self.fig.savefig('environment_example.png')  # Save as PNG
        if self.render_mode == 'human':
            plt.draw(); plt.pause(0.1)
            
            return  
        
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image: np.ndarray = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            return image

    def detect_tile_coords(self, num_tile: int, num_tilings: int, n_tiles_width: int, 
                          n_tiles_height: int, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Detects the coordinates of a tile in a tiled representation of the environment.

        Args:
            num_tile: The index of the tile.
            num_tilings: The number of tilings.
            n_tiles_width: The number of tiles along the width of the environment.
            n_tiles_height: The number of tiles along the height of the environment.
            pos: The (x, y) position in the environment.

        Returns:
            The (x, y) coordinates of the detected tile.
        """
        
        pos_x: float
        pos_y: float
        pos_x, pos_y = pos
        
        pos_x_norm: float = pos_x / self.width * n_tiles_width
        pos_y_norm: float = pos_y / self.height * n_tiles_height
        pos_x_shifted: float = pos_x_norm - 3*num_tile / num_tilings
        pos_y_shifted: float = pos_y_norm - num_tile / num_tilings
        
        tile_x: float = np.floor(pos_x_shifted) + 3*num_tile / num_tilings
        tile_y: float = np.floor(pos_y_shifted) + num_tile / num_tilings
        
        return tile_x * self.width / n_tiles_width, tile_y * self.width / n_tiles_width

    def _render_tiles(self, mode: str = 'human', n_tiles_width: int = 10, 
                     n_tiles_height: int = 10, n_tilings: int = 8) -> Optional[np.ndarray]:
        """Render the navigation environment with debug tiles.
        
        Args:
            mode: The mode of rendering. Default is 'human'. If 'rgb_array', returns an RGB array.
            n_tiles_width: Number of tiles along the width of the environment. Default is 10.
            n_tiles_height: Number of tiles along the height of the environment. Default is 10.
            n_tilings: Number of tilings for the tile coding. Default is 8.
            
            Returns:
            If mode is 'rgb_array', returns an RGB array of the rendered image.
        """
        
        tile_width: float = self.width / n_tiles_width
        tile_height: float = self.height / n_tiles_height
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')

        # Draw obstacles
        for obstacle in self.obstacles:
            self.ax.add_patch(Rectangle(obstacle[:2], obstacle[2], obstacle[3], fill=False, edgecolor='brown'))

        # Draw target area
        self.ax.add_patch(Rectangle(self.target_area[:2], self.target_area[2], self.target_area[3], 
                                    fill=True, facecolor='lightgreen', edgecolor='green', alpha=0.5))

        # Draw agent
        agent_color: str = 'orange'
        self.ax.add_patch(Circle(self.agent_pos, radius=self.agent_radius, fill=True, facecolor=agent_color))

        # Highlight active tiles
        active_tile_coords: List[Tuple[float, float]] = []
        
        for i in range(n_tilings):
            active_tile_coords.append(self.detect_tile_coords(i, n_tilings, n_tiles_width, n_tiles_height, self.agent_pos))
        
        for coord in active_tile_coords:
            # The coord might need to be decoded based on how it's stored in the IHT
            # This is a simplistic interpretation and might need adjustment
            active_tile: Rectangle = Rectangle(coord, tile_width, tile_height, edgecolor='black', facecolor='yellow', alpha=0.3)
            self.ax.add_patch(active_tile)

        plt.title('Navigation Environment')
        plt.draw()
        # self.fig.savefig('example.png')  # Save as PNG
        plt.pause(0.1)

        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image: np.ndarray = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            return image
        
        # Add grid labels
        # for x in range(n_tiles_width):
        #     self.ax.text(x * tile_width + tile_width/2, -0.3, str(x), 
        #                 ha='center', va='center', fontsize=8)
        # for y in range(n_tiles_height):
        #     self.ax.text(-0.3, y * tile_height + tile_height/2, str(y), 
        #                 ha='center', va='center', fontsize=8)

    def close(self) -> None:
        """Close the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    env: Navigation = Navigation()
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()
    done: bool = False

    for _ in range(20):  # Run for 20 steps
        action: int = env.action_space.sample()  # The agent takes a decission here
        
        reward: float
        terminated: bool
        truncated: bool
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f'Action: {action}; Observation: {obs}; done? {terminated or truncated}; Reward: {reward}')
        # env._render_tiles(n_tiles_width = 10, n_tiles_height = 10, n_tilings = 8) 
        env._render_tiles(mode='human', n_tiles_width = 10, n_tiles_height = 10, n_tilings = 8)
        
        if done:
            obs, info = env.reset()

    env.close()

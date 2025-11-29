import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import Optional, Tuple, List, Dict, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class WarehouseEnv(gym.Env):
    """Warehouse environment for reinforcement learning with object pickup and delivery tasks.
    
    Attributes:
        render_mode: The rendering mode ('human', 'rgb_array', or None).
        width: Width of the warehouse environment.
        height: Height of the warehouse environment.
        just_pick: If True, episode terminates after picking an object. If False, requires delivery.
        random_objects: If True, objects spawn at random positions on shelves.
        action_space: The discrete action space (Up, Down, Left, Right, Pick, Drop).
        observation_space: The continuous observation space.
        shelves: List of shelves defined by (x, y, width, height).
        delivery_area: The delivery area defined by (x, y, width, height).
        agent_radius: Radius of the agent.
        agent_velocity: Velocity of the agent.
        pickup_distance: Distance within which the agent can pick up objects.
        fig: Figure for rendering.
        ax: Axis for rendering.
        agent_pos: Current position of the agent.
        object_positions: Positions of objects in the environment.
        agent_has_object: Whether the agent is currently holding an object.
        delivery: Whether the agent successfully delivered an object.
        collision: Whether a collision has occurred.
        steps: Number of steps taken in current episode.
        max_steps: Maximum number of steps allowed per episode.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, just_pick: bool = True, random_objects: bool = False, 
                 render_mode: Optional[str] = None) -> None:
        """Initialize the WarehouseEnv environment.
        
        Args:
            just_pick: If True, episode ends after picking an object. Defaults to True.
            random_objects: If True, objects spawn randomly on shelves. Defaults to False.
            render_mode: The rendering mode ('human', 'rgb_array', or None). Defaults to None.
        """
        
        super().__init__()

        self.render_mode: Optional[str] = render_mode
        self.width: float = 10.0
        self.height: float = 10.0

        self.just_pick: bool = just_pick
        self.random_objects: bool = random_objects

        # Define action and observation spaces
        n_actions: int = 5 if self.just_pick else 6
        self.action_space: spaces.Discrete = spaces.Discrete(n_actions)
        self.observation_space: spaces.Box = spaces.Box(low=0, high=10, shape=(11,), dtype=np.float32)

        # Warehouse layout
        self.shelves: List[Tuple[float, float, float, float]] = [
            (1.9, 1.0, 0.2, 5.0), 
            (4.9, 1.0, 0.2, 5.0), 
            (7.9, 1.0, 0.2, 5.0)
        ]
        self.delivery_area: Tuple[float, float, float, float] = (2.5, 9, 5.0, 2.0)

        # Agent properties
        self.agent_radius: float = 0.2
        self.agent_velocity: float = 0.25
        self.pickup_distance: float = 0.6

        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

        self.reset()

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.
        
        Args:
            options: Optional configuration for resetting the environment.
            
        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        
        super().reset()

        self.agent_pos: Tuple[float, float] = self._get_random_empty_position()

        if self.random_objects:
            self.object_positions: List[Optional[Tuple[float, float]]] = [
                self._get_random_position_on_shelf(s) for s in self.shelves
            ]
        else:
            self.object_positions: List[Optional[Tuple[float, float]]] = [
                (2, 3.0), (8, 4.0), (5, 2.0)
            ]

        self.agent_has_object: bool = False
        self.delivery: bool = False
        self.collision: bool = False

        self.steps: int = 0
        self.max_steps: int = 20000

        obs: np.ndarray = self._get_obs()
        info: Dict[str, Any] = {}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Perform a step in the environment based on the given action.
        
        Args:
            action: The action to be taken by the agent.
                0 - Move Up
                1 - Move Down
                2 - Move Left
                3 - Move Right
                4 - Pick up object
                5 - Drop object (only available if just_pick=False)
            
        Returns:
            A tuple containing:
                - observation: The observation after taking the action.
                - reward: The reward received.
                - terminated: Whether the episode has terminated.
                - truncated: Whether the episode was truncated.
                - info: Additional information about the step.
        """
        
        # --- Reward definition ---
        REWARD_TO_BE_DESIGNED: float = 0.0

        # --- Initialize state variables ---
        self.steps += 1
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False

        # --- Action handling ---
        if action < 4:  # Movement
            new_pos: Tuple[float, float] = self._get_new_position(action)
            if not self._is_collision(new_pos):
                self.agent_pos = new_pos
                reward = max([-1 + 2.71828**(-2*self._distance(self.agent_pos, obj_pos)/(self.width**2 + self.height**2)**0.5) for obj_pos in self.object_positions])
            else:
                self.collision = True
                terminated = True
                reward = -1

        elif action == 4:  # Pick
            if not self.agent_has_object:
                for i, obj_pos in enumerate(self.object_positions):
                    if obj_pos is not None and self._distance(self.agent_pos, obj_pos) <= self.pickup_distance + self.agent_radius:
                        self.agent_has_object = True
                        self.object_positions[i] = None
                        reward = 1
                        if self.just_pick:
                            terminated = True
                        break

        elif action == 5:  # Drop
            if self.agent_has_object:
                if self._is_in_area(self.agent_pos, self.delivery_area):
                    reward = 1
                    self.delivery = True
                else:
                    reward = max([-1 + 2.71828**(-2*self._distance(self.agent_pos, obj_pos)/(self.width**2 + self.height**2)**0.5) for obj_pos in self.object_positions])
                    self.object_positions.append(self.agent_pos)
                self.agent_has_object = False
                terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        # --- Prepare outputs ---
        obs: np.ndarray = self._get_obs()
        info: Dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get the current observation.
        
        Returns:
            The current observation array containing:
                - Agent position (x, y)
                - Three object positions (x, y) each, or agent position if object was picked
                - Agent has object flag
                - Collision flag
                - Delivery flag
        """
        
        obs: np.ndarray = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        for i, obj in enumerate(self.object_positions):
            if obj is not None:
                obs[2 + 2 * i: 4 + 2 * i] = obj
            else:
                obs[2 + 2 * i: 4 + 2 * i] = self.agent_pos
        obs[8] = float(self.agent_has_object)
        obs[9] = float(self.collision)
        obs[10] = float(self.delivery)
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
        
        x: float
        y: float
        x, y = self.agent_pos
        
        if action == 0:  # Up
            y = min(self.height - self.agent_radius, y + self.agent_velocity)
        elif action == 1:  # Down
            y = max(self.agent_radius, y - self.agent_velocity)
        elif action == 2:  # Left
            x = max(self.agent_radius, x - self.agent_velocity)
        elif action == 3:  # Right
            x = min(self.width - self.agent_radius, x + self.agent_velocity)
        
        return (x, y)

    def _is_collision(self, pos: Tuple[float, float]) -> bool:
        """Check if the given position results in a collision.
        
        Args:
            pos: A tuple representing the (x, y) coordinates of the position to check.
            
        Returns:
            True if there is a collision with walls or shelves, False otherwise.
        """
        
        # Check for collisions with walls
        if (
            pos[0] <= self.agent_radius or
            pos[0] >= self.width - self.agent_radius or
            pos[1] <= self.agent_radius or
            pos[1] >= self.height - self.agent_radius
        ):
            return True

        # Check for collisions with shelves
        for shelf in self.shelves:
            if self._is_in_area(pos, shelf, self.agent_radius):
                return True

        return False

    def _get_random_empty_position(self) -> Tuple[float, float]:
        """Generate a random position within the environment that does not collide with any obstacles.

        Returns:
            A tuple (x, y) representing the coordinates of a random, collision-free position.
        """
        
        while True:
            pos: Tuple[float, float] = (
                np.random.uniform(self.agent_radius, self.width - self.agent_radius),
                np.random.uniform(self.agent_radius, self.height - self.agent_radius),
            )
            if not self._is_collision(pos):
                return pos

    def _get_random_position_on_shelf(self, shelf: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Generate a random position on the given shelf.

        Args:
            shelf: A tuple representing the shelf with the format (x, y, width, height).

        Returns:
            A tuple (x, y) representing the random position on the shelf.
        """
        
        aux: float = np.random.uniform(0, 1)
        x: float = shelf[0] + (0.25 if aux < 0.5 else 0.75) * shelf[2]
        y: float = np.random.uniform(shelf[1] + 0.5, shelf[1] + shelf[3] - 0.5)
        return (x, y)

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points.

        Args:
            a: A tuple representing the (x, y) coordinates of the first point.
            b: A tuple representing the (x, y) coordinates of the second point.

        Returns:
            The Euclidean distance between the two points.
        """
        
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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
        
        return (
            area[0] - margin <= pos[0] <= area[0] + area[2] + margin and
            area[1] - margin <= pos[1] <= area[1] + area[3] + margin
        )

    def render(self) -> Optional[np.ndarray]:
        """Renders the current state of the warehouse environment.
        
        Returns:
            If render_mode is 'rgb_array', returns an RGB array of the rendered image.
            Otherwise, returns None.
        """
        
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")

        # Shelves
        for s in self.shelves:
            self.ax.add_patch(Rectangle(s[:2], s[2], s[3], fill=False, edgecolor="brown"))

        # Delivery area
        self.ax.add_patch(Rectangle(self.delivery_area[:2], self.delivery_area[2], self.delivery_area[3],
                                    fill=True, facecolor="lightgreen", edgecolor="green", alpha=0.5))

        # Objects
        for obj in self.object_positions:
            if obj is not None:
                self.ax.add_patch(Circle(obj, radius=0.2, color="blue"))

        # Agent
        color: str = "red" if self.agent_has_object else "orange"
        self.ax.add_patch(Circle(self.agent_pos, radius=self.agent_radius, color=color))

        plt.title("WarehouseEnv")
        plt.draw()
        plt.pause(0.05)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image: np.ndarray = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self) -> None:
        """Close the rendering window."""
        
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None


if __name__ == "__main__":
    env: WarehouseEnv = WarehouseEnv(just_pick=True, random_objects=False, render_mode="human")
    obs: np.ndarray
    info: Dict[str, Any]
    obs, info = env.reset()
    
    for _ in range(100):
        action: int = env.action_space.sample()
        reward: float
        terminated: bool
        truncated: bool
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

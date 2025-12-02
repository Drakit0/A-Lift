import numpy as np
from tiles3 import IHT, tiles
from typing import List, Tuple

class FeedbackConstruction:
    """Feedback construction using tile coding for state representation.
    
    Attributes:
        width: Width of the environment.
        height: Height of the environment.
        scale_width: Scale factor for width normalization.
        scale_height: Scale factor for height normalization.
        target_area: The target area coordinates.
        num_tilings: Number of tilings for tile coding.
        max_size: Maximum size of the index hash table.
        iht: Index hash table for tile coding.
    """
    
    def __init__(self, dims: Tuple[float, float] = (10.0, 10.0), n_tiles: Tuple[int, int] = (10, 10), 
                n_tilings: int = 8, target_area: Tuple[float, float, float, float] = (2.5, 8, 1.0, 2.0), 
                use_tiles: bool = True, just_pick: bool = True, random_objects: bool = False,
                use_rich_features: bool = False) -> None:
        """Initialize the FeedbackConstruction.
        
        Args:
            dims: Tuple of (width, height) of the environment.
            n_tiles: Tuple of (n_tiles_width, n_tiles_height) for tiling.
            n_tilings: Number of tilings for tile coding.
            target_area: The target area as (x, y, width, height).
            use_tiles: Whether to use tile coding (True) or vectorized representation (False).
            just_pick: If True, env 1 (just pick). If False, env 2 or 3 (pick and deliver).
            random_objects: If True, env 3 (random object positions).
            use_rich_features: If True, use enhanced feature engineering for DQN (23 features).
        """
        
        self.width: float = dims[0]
        self.height: float = dims[1]
        self.target_area: Tuple[float, float, float, float] = target_area
        self.use_tiles = use_tiles
        self.just_pick = just_pick
        self.random_objects = random_objects
        self.use_rich_features = use_rich_features
        
        # For rich features: max diagonal distance for normalization
        self.distance_scale = np.sqrt(dims[0]**2 + dims[1]**2)
        
        # Shelves layout for obstacle proximity
        self.shelves = [(1.9, 1.0, 0.2, 5.0), (4.9, 1.0, 0.2, 5.0), (7.9, 1.0, 0.2, 5.0)]
        self.agent_radius = 0.2

        if self.use_tiles:
            self.scale_width: float = dims[0] / n_tiles[0]
            self.scale_height: float = dims[1] / n_tiles[1]  
            self.num_tilings: int = n_tilings
            # Increase max_size for env 2 and 3 to accommodate direction and distance bins
            if just_pick:
                extra_capacity = 2000  # Env 1: simpler state space
            else:
                extra_capacity = 8000  # Env 2 & 3: direction + distance bins
            self.max_size: int = n_tiles[0] * n_tiles[1] * self.num_tilings + extra_capacity
            self.iht: IHT = IHT(self.max_size)
            self.observation_size: int = self.iht.size
        else:
            if self.use_rich_features:
                self.observation_size: int = 23  # Rich features for DQN
            else:
                self.observation_size: int = 11  # Raw observation
            self.iht: IHT = None

        
    def process_observation(self, obs: np.ndarray, dense_vector: bool = False) -> np.ndarray:
        """Processes the environment observation and returns the active tile features.
        
        Args:
            obs: Observation from the environment containing at least four elements:
                - obs[0:2]: agent (x, y) position in environment coordinates.
                - obs[8]: agent has object
            dense_vector: requires the function dense vectors for NNs.
                
        Returns:
            Array of active tiles as produced by _get_active_tiles, or feature vector.
        """
        
        if self.use_tiles and not dense_vector:
            agent_pos: np.ndarray = obs[:2]
            has_object: int = 1 if obs[8] > 0.5 else 0
            
            # Normalize agent position
            norm_x: float = agent_pos[0] / self.scale_width
            norm_y: float = agent_pos[1] / self.scale_height
            
            if self.just_pick: # Env 1
                active_tiles: List[int] = self._get_active_tiles(norm_x, norm_y, has_object)
                
            else: # Env 2 or Env 3
                if has_object == 0: # Dir to obj
                    target = self._get_nearest_object(agent_pos, obs)
                    
                else: # Dir to unloading zone
                    target = (self.target_area[0] + self.target_area[2]/2,
                              self.target_area[1] + self.target_area[3]/2)
                
                # Discretize direction to target (8 directions + at target)
                direction: int = self._get_direction(agent_pos, target)
                distance_bin: int = self._get_distance_bin(agent_pos, target)
                
                active_tiles: List[int] = self._get_active_tiles_more_info(norm_x, norm_y, has_object, direction, distance_bin)
            
            return active_tiles
        
        # DQN with rich features 
        if self.use_rich_features:
            return self._process_rich_features(obs)
        
        # obs expected length 11: [agent_x, agent_y, obj1_x,obj1_y, obj2_x,obj2_y, obj3_x,obj3_y, has_object, collision, delivery]
        vec = np.array(obs, dtype=np.float32).copy()

        # Normalize x coordinates (indices 0,2,4,6) by width
        vec[0] = vec[0] / (self.width if self.width != 0 else 1.0)
        vec[2] = vec[2] / (self.width if self.width != 0 else 1.0)
        vec[4] = vec[4] / (self.width if self.width != 0 else 1.0)
        vec[6] = vec[6] / (self.width if self.width != 0 else 1.0)

        # Normalize y coordinates (indices 1,3,5,7) by height
        vec[1] = vec[1] / (self.height if self.height != 0 else 1.0)
        vec[3] = vec[3] / (self.height if self.height != 0 else 1.0)
        vec[5] = vec[5] / (self.height if self.height != 0 else 1.0)
        vec[7] = vec[7] / (self.height if self.height != 0 else 1.0)

        # flags at indices 8,9,10 are already 0/1 floats
        return vec
    
    def _process_rich_features(self, obs: np.ndarray) -> np.ndarray:
        """Process observation into rich engineered features for DQN.
        
        Creates a 23-dimensional feature vector with:
        - Normalized agent position (2)
        - Has object flag (1)
        - Distances to each object normalized (3)
        - Distance to nearest object (1)
        - Direction to nearest object (2)
        - Distance to delivery zone (1)
        - Direction to delivery zone (2)
        - Relative positions to objects (6)
        - Obstacle proximity in 4 directions (4)
        - Near pickup flag (1)
        
        Args:
            obs: Raw observation from environment.
            
        Returns:
            23-dimensional feature vector.
        """
        agent_pos = obs[0:2]
        obj1_pos = obs[2:4]
        obj2_pos = obs[4:6]
        obj3_pos = obs[6:8]
        has_object = obs[8]
        
        features = []
        
        # 1. Normalized agent position (2 features)
        features.extend(agent_pos / self.width)  # Assuming square environment
        
        # 2. Has object flag (1 feature)
        features.append(has_object)
        
        # 3. Distances to each object normalized (3 features)
        dist_obj1 = self._distance(agent_pos, obj1_pos) / self.distance_scale
        dist_obj2 = self._distance(agent_pos, obj2_pos) / self.distance_scale
        dist_obj3 = self._distance(agent_pos, obj3_pos) / self.distance_scale
        features.extend([dist_obj1, dist_obj2, dist_obj3])
        
        # 4. Distance to nearest object (1 feature)
        min_dist = min(dist_obj1, dist_obj2, dist_obj3)
        features.append(min_dist)
        
        # 5. Direction to nearest object (2 features)
        if dist_obj1 <= dist_obj2 and dist_obj1 <= dist_obj3:
            closest_obj = obj1_pos
        elif dist_obj2 <= dist_obj3:
            closest_obj = obj2_pos
        else:
            closest_obj = obj3_pos
        
        direction_obj = self._unit_vector(agent_pos, closest_obj)
        features.extend(direction_obj)
        
        # 6. Distance to delivery zone (1 feature)
        delivery_center = np.array([
            self.target_area[0] + self.target_area[2] / 2,
            self.target_area[1] + self.target_area[3] / 2
        ])
        dist_delivery = self._distance(agent_pos, delivery_center) / self.distance_scale
        features.append(dist_delivery)
        
        # 7. Direction to delivery zone (2 features)
        direction_delivery = self._unit_vector(agent_pos, delivery_center)
        features.extend(direction_delivery)
        
        # 8. Relative positions to objects (6 features)
        # If agent doesn't have object, these help locate objects
        if has_object == 0:
            features.extend((obj1_pos - agent_pos) / self.width)
            features.extend((obj2_pos - agent_pos) / self.width)
            features.extend((obj3_pos - agent_pos) / self.width)
        else:
            # If has object, use delivery direction info instead
            features.extend((delivery_center - agent_pos) / self.width)
            features.extend([0.0, 0.0])  # Padding
            features.extend([0.0, 0.0])  # Padding
        
        # 9. Obstacle proximity in 4 directions (4 features)
        obstacle_proximity = self._get_obstacle_proximity(agent_pos)
        features.extend(obstacle_proximity)
        
        # 10. Near pickup flag - can pick up an object (1 feature)
        pickup_threshold = 0.8  # Agent radius + pickup distance
        near_pickup = float(min_dist * self.distance_scale < pickup_threshold)
        features.append(near_pickup)
        
        return np.array(features, dtype=np.float32)
    
    def _get_obstacle_proximity(self, agent_pos: np.ndarray) -> List[float]:
        """Calculate proximity to obstacles in 4 cardinal directions.
        
        Args:
            agent_pos: Current agent position.
            
        Returns:
            List of 4 proximity values (0=far, 1=very close) for [up, down, left, right].
        """
        velocity = 0.25  # Agent velocity
        max_dist = 2.0   # Max distance to check
        
        # Check 4 directions
        directions = [
            (0, velocity),   # Up
            (0, -velocity),  # Down
            (-velocity, 0),  # Left
            (velocity, 0)    # Right
        ]
        
        proximity = []
        for dx, dy in directions:
            min_dist = self._min_obstacle_distance(
                agent_pos[0], agent_pos[1], (dx, dy), velocity, max_dist
            )
            # Convert to proximity: 0 = far, 1 = close
            prox = 1.0 - (min_dist / max_dist)
            proximity.append(max(0.0, min(1.0, prox)))
        
        return proximity
    
    def _min_obstacle_distance(self, x: float, y: float, direction: Tuple[float, float], 
                                velocity: float, max_dist: float) -> float:
        """Calculate minimum distance to obstacle in given direction.
        
        Args:
            x, y: Current position.
            direction: Direction vector (dx, dy).
            velocity: Step size.
            max_dist: Maximum distance to check.
            
        Returns:
            Distance to nearest obstacle in that direction.
        """
        dist = 0.0
        dx, dy = direction
        
        # Normalize direction
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude == 0:
            return max_dist
        dx, dy = dx / magnitude, dy / magnitude
        
        step = velocity
        while dist < max_dist:
            test_x = x + dx * dist
            test_y = y + dy * dist
            
            if self._would_collide(test_x, test_y):
                return dist
            
            dist += step
        
        return max_dist
    
    def _would_collide(self, x: float, y: float) -> bool:
        """Check if position would cause collision.
        
        Args:
            x, y: Position to check.
            
        Returns:
            True if collision would occur.
        """
        # Wall collision
        if (x <= self.agent_radius or x >= self.width - self.agent_radius or
            y <= self.agent_radius or y >= self.height - self.agent_radius):
            return True
        
        # Shelf collision
        for shelf in self.shelves:
            sx, sy, sw, sh = shelf
            if (sx - self.agent_radius <= x <= sx + sw + self.agent_radius and
                sy - self.agent_radius <= y <= sy + sh + self.agent_radius):
                return True
        
        return False
    
    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.sqrt(np.sum((a - b) ** 2)))
    
    @staticmethod
    def _unit_vector(from_pos: np.ndarray, to_pos: np.ndarray) -> List[float]:
        """Calculate unit vector from one position to another.
        
        Args:
            from_pos: Starting position.
            to_pos: Target position.
            
        Returns:
            Unit vector [dx, dy], or [0, 0] if positions are the same.
        """
        diff = to_pos - from_pos
        magnitude = np.sqrt(np.sum(diff ** 2))
        
        if magnitude < 1e-6:
            return [0.0, 0.0]
        
        return [float(diff[0] / magnitude), float(diff[1] / magnitude)]
    
    def _get_nearest_object(self, agent_pos: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
        """Get position of nearest available object.
        
        Args:
            agent_pos: Current agent position.
            obs: Full observation array.
            
        Returns:
            Position (x, y) of nearest object.
        """
        min_dist: float = float('inf')
        nearest: Tuple[float, float] = (agent_pos[0], agent_pos[1])
        
        for i in range(3):
            obj_x = obs[2 + 2*i]
            obj_y = obs[3 + 2*i]
            
            # Skip if object position equals agent position (already picked)
            if abs(obj_x - agent_pos[0]) < 0.01 and abs(obj_y - agent_pos[1]) < 0.01:
                continue
            
            dist = np.sqrt((agent_pos[0] - obj_x)**2 + (agent_pos[1] - obj_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest = (obj_x, obj_y)
        
        return nearest
    
    def _get_direction(self, agent_pos: np.ndarray, target: Tuple[float, float]) -> int:
        """Get discretized direction from agent to target.
        
        Args:
            agent_pos: Current agent position.
            target: Target position (x, y).
            
        Returns:
            Direction index (0-8, where 8 means at target).
        """
        dx = target[0] - agent_pos[0]
        dy = target[1] - agent_pos[1]
        
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 0.5:  # At target
            return 8
        
        angle = np.arctan2(dy, dx)
        direction = int((angle + np.pi) / (2 * np.pi) * 8) % 8
        return direction
    
    def _get_distance_bin(self, agent_pos: np.ndarray, target: Tuple[float, float]) -> int:
        """Get discretized distance from agent to target.
        
        Args:
            agent_pos: Current agent position.
            target: Target position (x, y).
            
        Returns:
            Distance bin (0-4): 0=very close, 1=close, 2=medium, 3=far, 4=very far
        """
        dx = target[0] - agent_pos[0]
        dy = target[1] - agent_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Normalize by diagonal of environment
        max_dist = np.sqrt(self.width**2 + self.height**2)
        norm_dist = dist / max_dist
        
        if norm_dist < 0.1:
            return 0  # very close
        
        elif norm_dist < 0.25:
            return 1  # close
        
        elif norm_dist < 0.5:
            return 2  # medium
        
        elif norm_dist < 0.75:
            return 3  # far
        
        else:
            return 4  # very far

    def _get_active_tiles(self, norm_x: float, norm_y: float, has_object: int = 0) -> List[int]:
        """Calculate the active tiles for given normalized x and y coordinates.
        
        Args:
            norm_x: Normalized x-coordinate.
            norm_y: Normalized y-coordinate.
            has_object: the agent has an object
            
        Returns:
            A list of active tile indices.
        """
        
        # Implementation of the tiling with odd offset (3, 1)                        
        offset_factor_x: float = 1/self.num_tilings * 3
        offset_factor_y: float = 1/self.num_tilings * 1
        active_tiles: List[int] = []
        
        for i in range(self.num_tilings):
            offset_x: float = offset_factor_x * i
            offset_y: float = offset_factor_y * i
            
            tile_temp: List[int] = tiles(self.iht, 1, 
                    [norm_x - offset_x, 
                    norm_y - offset_y],
                    ints=[i, has_object])
            
            active_tiles.append(tile_temp[0])
                
        return active_tiles
    
    def _get_active_tiles_more_info(self, norm_x: float, norm_y: float, 
                                has_object: int, direction: int, distance_bin: int) -> List[int]:
        """Calculate active tiles for Env 2 and 3 including direction and distance information.
        
        Args:
            norm_x: Normalized x-coordinate.
            norm_y: Normalized y-coordinate.
            has_object: Whether agent holds an object (0 or 1).
            direction: Discretized direction to target (0-8).
            distance_bin: Discretized distance to target (0-4).
            
        Returns:
            List of active tile indices.
        """
        offset_factor_x: float = 1/self.num_tilings * 3
        offset_factor_y: float = 1/self.num_tilings * 1
        active_tiles: List[int] = []
        
        for i in range(self.num_tilings):
            offset_x: float = offset_factor_x * i
            offset_y: float = offset_factor_y * i
            
            tile_temp: List[int] = tiles(self.iht, 1, 
                    [norm_x - offset_x, 
                     norm_y - offset_y],
                    ints=[i, has_object, direction, distance_bin])
            
            active_tiles.append(tile_temp[0])
                
        return active_tiles
  
if __name__ == "__main__":

    warehouse_width: float = 10.0
    warehouse_height: float = 10.0
    target_area: Tuple[float, float, float, float] = (2.5, 8, 5.0, 2.0)

    # Start the experiment
    n_tiles_width: int = 1
    n_tiles_height: int = 1
    n_tilings: int = 2

    feedback: FeedbackConstruction = FeedbackConstruction((warehouse_width, warehouse_height), 
                                                          (n_tiles_width, n_tiles_height), 
                                                          n_tilings, 
                                                          target_area)

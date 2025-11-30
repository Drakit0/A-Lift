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
                use_tiles: bool = True, just_pick: bool = True) -> None:
        """Initialize the FeedbackConstruction.
        
        Args:
            dims: Tuple of (width, height) of the environment.
            n_tiles: Tuple of (n_tiles_width, n_tiles_height) for tiling.
            n_tilings: Number of tilings for tile coding.
            target_area: The target area as (x, y, width, height).
        """
        
        self.width: float = dims[0]
        self.height: float = dims[1]
        self.target_area: Tuple[float, float, float, float] = target_area
        self.use_tiles = use_tiles
        self.just_pick = just_pick

        if self.use_tiles:
            self.scale_width: float = dims[0] / n_tiles[0]
            self.scale_height: float = dims[1] / n_tiles[1]  
            self.num_tilings: int = n_tilings
            self.max_size: int = n_tiles[0] * n_tiles[1] * self.num_tilings + 4000
            self.iht: IHT = IHT(self.max_size)
            self.observation_size:int = self.iht.size
        else:
            self.observation_size:int = 11
            self.iht: IHT = None

        
    def process_observation(self, obs: np.ndarray, dense_vector: bool = False) -> List[int]:
        """Processes the environment observation and returns the active tile features.
        
        Args:
            obs: Observation from the environment containing at least four elements:
                - obs[0:2]: agent (x, y) position in environment coordinates.
                - obs[8]: agent has object
            dense_vector: requires the function dense vectors for NNs.
                
        Returns:
            Array of active tiles as produced by _get_active_tiles.
        """
        
        if self.use_tiles and not dense_vector:
            agent_pos: np.ndarray = obs[:2]
            has_object: int = 1 if obs[8] > 0.5 else 0
            
            # Normalize agent position
            norm_x: float = agent_pos[0] / self.scale_width
            norm_y: float = agent_pos[1] / self.scale_height
            
            if self.just_pick: # Env 1
                active_tiles: List[int] = self._get_active_tiles(norm_x, norm_y, has_object)
                
            else: # Env 2
                if has_object == 0: # Dir to obj
                    target = self._get_nearest_object(agent_pos, obs)
                    
                else: # Dir to unloading zone
                    target = (self.target_area[0] + self.target_area[2]/2,
                              self.target_area[1] + self.target_area[3]/2)
                
                # Discretize direction to target (8 directions + at target)
                direction: int = self._get_direction(agent_pos, target)
                active_tiles: List[int] = self._get_active_tiles_env23(norm_x, norm_y, has_object, direction)
            
            return active_tiles
        
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
        if dist < 0.1:  # At target 
            return 8
        
        angle = np.arctan2(dy, dx)
        direction = int((angle + np.pi) / (2 * np.pi) * 8) % 8
        return direction

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
    
    def _get_active_tiles_env23(self, norm_x: float, norm_y: float, 
                                has_object: int, direction: int) -> List[int]:
        """Calculate active tiles for Env 2/3 including direction information.
        
        Args:
            norm_x: Normalized x-coordinate.
            norm_y: Normalized y-coordinate.
            has_object: Whether agent holds an object (0 or 1).
            direction: Discretized direction to target (0-8).
            
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
                    ints=[i, has_object, direction])
            
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

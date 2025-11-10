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
    
    def __init__(self, dims: Tuple[float, float], n_tiles: Tuple[int, int], 
                 n_tilings: int, target_area: Tuple[float, float, float, float]) -> None:
        """Initialize the FeedbackConstruction.
        
        Args:
            dims: Tuple of (width, height) of the environment.
            n_tiles: Tuple of (n_tiles_width, n_tiles_height) for tiling.
            n_tilings: Number of tilings for tile coding.
            target_area: The target area as (x, y, width, height).
        """
        
        self.width: float = dims[0]
        self.height: float = dims[1]
        self.scale_width: float = dims[0] / n_tiles[0]
        self.scale_height: float = dims[1] / n_tiles[1]
        self.target_area: Tuple[float, float, float, float] = target_area        
        self.num_tilings: int = n_tilings
        self.max_size: int = n_tiles[0] * n_tiles[1] * self.num_tilings + 2000
        self.iht: IHT = IHT(self.max_size)

        
    def process_observation(self, obs: np.ndarray) -> List[int]:
        """Processes the environment observation and returns the active tile features.
        
        Args:
            obs: Observation from the environment containing at least four elements:
                - obs[0:2]: agent (x, y) position in environment coordinates.
                - obs[2]: collision flag (read but not used by this implementation).
                - obs[3]: target area identifier (read but not used by this implementation).
                
        Returns:
            Array of active tiles as produced by _get_active_tiles.
        """
        
        agent_pos: np.ndarray = obs[:2]
        collision: float = obs[2]
        target_area: float = obs[3]
        
        # Normalize agent position
        norm_x: float = agent_pos[0] / self.scale_width
        norm_y: float = agent_pos[1] / self.scale_height
        
        # Get active tiles
        active_tiles: List[int] = self._get_active_tiles(norm_x, norm_y)

        observation: List[int] = active_tiles

        return observation

    def _get_active_tiles(self, norm_x: float, norm_y: float) -> List[int]:
        """Calculate the active tiles for given normalized x and y coordinates.
        
        Args:
            norm_x: Normalized x-coordinate.
            norm_y: Normalized y-coordinate.
            
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
                    ints=[i])
            
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

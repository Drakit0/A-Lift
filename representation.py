import numpy as np
from tiles3 import IHT, tiles

class FeedbackConstruction:
    def __init__(self, dims, n_tiles, n_tilings, target_area):
        # Don't touch these lines
        self.width = dims[0]
        self.height = dims[1]
        self.scale_width = dims[0] / n_tiles[0]
        self.scale_height = dims[1] / n_tiles[1]
        self.target_area = target_area        
        self.num_tilings = n_tilings
        self.max_size = n_tiles[0] * n_tiles[1] * self.num_tilings + 2000
        self.iht = IHT(self.max_size)
        ##############################
        # If you want to add more attributes, add them from here
        
    def process_observation(self, obs):
        """
        Processes the environment observation and returns the active tile features corresponding to the agent's normalized position.
            obs (Sequence or np.ndarray): Observation from the environment containing at least four elements:
                - obs[0:2]: agent (x, y) position in environment coordinates.
                - obs[2]: collision flag (read but not used by this implementation).
                - obs[3]: target area identifier (read but not used by this implementation).
            np.ndarray: Array of active tiles as produced by self._get_active_tiles(norm_x, norm_y).
                        The agent position is normalized by self.scale_width and self.scale_height
                        prior to computing active tiles.
        Notes:
            - This method expects the instance to provide numeric attributes scale_width and scale_height,
              and a method _get_active_tiles(norm_x, norm_y) that maps normalized coordinates to tile features.
            - Only the agent position is currently used to compute the returned observation; other components
              of obs (collision, target area) are read but not incorporated into the returned value.
        """
        agent_pos = obs[:2]
        collision = obs[2]
        target_area = obs[3]
        
        # Normalize agent position
        norm_x = agent_pos[0] / self.scale_width
        norm_y = agent_pos[1] / self.scale_height
        
        # Get active tiles
        active_tiles = self._get_active_tiles(norm_x, norm_y)

        # Add your code here to return the processed observation
        observation = active_tiles
        ##############################

        return observation

    def _get_active_tiles(self, norm_x, norm_y):
        """
        Calculate the active tiles for given normalized x and y coordinates.
        This method computes the active tiles based on the normalized coordinates
        (norm_x, norm_y) and the number of tilings. It applies specific offsets
        for each tiling to determine the active tiles.
        Args:
            norm_x (float): Normalized x-coordinate.
            norm_y (float): Normalized y-coordinate.
        Returns:
            list: A list of active tile indices.
        """
        
        # Implementation of the tiling with odd offset (3, 1)                        
        offset_factor_x = 1/self.num_tilings * 3
        offset_factor_y = 1/self.num_tilings * 1
        active_tiles = []
        
        for i in range(self.num_tilings):
            offset_x = offset_factor_x * i
            offset_y = offset_factor_y * i
            
            tile_temp = tiles(self.iht, 1, 
                    [norm_x - offset_x, 
                    norm_y - offset_y],
                    ints=[i])
            
            active_tiles.append(tile_temp[0])
                
        return active_tiles

  
if __name__ == "__main__":
    # Space for testing

    # No need to touch this code
    warehouse_width = 10.0
    warehouse_height = 10.0
    target_area = (2.5, 8, 5.0, 2.0)
    ##############################
    # Complete freedom from here
    n_tiles_width = 1
    n_tiles_height = 1
    n_tilings = 2


    feedback = FeedbackConstruction((warehouse_width, warehouse_height), (n_tiles_width, n_tiles_height), n_tilings, 
                                target_area)

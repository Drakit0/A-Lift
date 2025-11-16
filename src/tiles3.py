"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
"""

from math import floor
from itertools import zip_longest
from typing import List, Dict, Tuple, Optional, Union

basehash = hash

class IHT:
    """Structure to handle collisions.
    
    Attributes:
        size: The maximum size of the hash table.
        overfullCount: Counter for collisions when table is full.
        dictionary: Dictionary mapping tile coordinates to indices.
    """
    
    def __init__(self, sizeval: int) -> None:
        """Initialize the IHT.
        
        Args:
            sizeval: The maximum size of the hash table.
        """
        
        self.size: int = sizeval                        
        self.overfullCount: int = 0
        self.dictionary: Dict[Tuple[int, ...], int] = {}

    def __str__(self) -> str:
        """Prepares a string for printing whenever this object is printed.
        
        Returns:
            A string representation of the collision table.
        """
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self) -> int:
        """Get the number of items in the dictionary.
        
        Returns:
            The number of items stored in the dictionary.
        """
        return len(self.dictionary)
    
    def fullp(self) -> bool:
        """Check if the table is full.
        
        Returns:
            True if the table is full, False otherwise.
        """
        return len(self.dictionary) >= self.size
    
    def getindex(self, obj: Tuple[int, ...], readonly: bool = False) -> Optional[int]:
        """Get the index for a given object.
        
        Args:
            obj: The object tuple to get the index for.
            readonly: If True, don't add new entries to the dictionary.
            
        Returns:
            The index for the object, or None if readonly and not found.
        """
        
        d: Dict[Tuple[int, ...], int] = self.dictionary
        
        if obj in d: 
            return d[obj]
        
        elif readonly: 
            return None
        
        size: int = self.size
        count: int = self.count()
        
        if count >= size:
            if self.overfullCount==0: 
                print('IHT full, starting to allow collisions')
                
            self.overfullCount += 1
            
            return basehash(obj) % self.size
        
        else:
            d[obj] = count
            return count

def hashcoords(coordinates: List[int], m: Union[IHT, int, None], readonly: bool = False) -> Union[int, List[int]]:
    """Hash tile coordinates to an index.
    
    Args:
        coordinates: List of tile coordinates.
        m: IHT object, integer size, or None.
        readonly: If True, don't add new entries.
        
    Returns:
        The hashed index or the coordinates themselves.
    """
    
    if type(m)==IHT: 
        return m.getindex(tuple(coordinates), readonly)
    
    if type(m)==int: 
        return basehash(tuple(coordinates)) % m
    
    if m==None: 
        return coordinates

def tiles(ihtORsize: Union[IHT, int, None], numtilings: int, floats: List[float], 
          ints: List[int] = [], readonly: bool = False) -> List[int]:
    """Returns num-tilings tile indices corresponding to the floats and ints.
    
    Args:
        ihtORsize: IHT object, integer size, or None.
        numtilings: Number of tilings.
        floats: List of float values to be tiled.
        ints: List of integer values to append to coordinates.
        readonly: If True, don't add new entries.
        
    Returns:
        List of tile indices.
    """
    
    qfloats: List[int] = [floor(f*numtilings) for f in floats]
    Tiles: List[int] = []
    
    for tiling in range(numtilings):
        tilingX2: int = tiling*2
        coords: List[int] = [tiling]
        b: int = tiling
        
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
            
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
        
    return Tiles

def tileswrap(ihtORsize: Union[IHT, int, None], numtilings: int, floats: List[float], 
              wrapwidths: List[Optional[int]], ints: List[int] = [], readonly: bool = False) -> List[int]:
    """Returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats.
    
    Args:
        ihtORsize: IHT object, integer size, or None.
        numtilings: Number of tilings.
        floats: List of float values to be tiled.
        wrapwidths: List of wrap widths for each float (None for no wrapping).
        ints: List of integer values to append to coordinates.
        readonly: If True, don't add new entries.
        
    Returns:
        List of tile indices with wrapping applied.
    """
    
    qfloats: List[int] = [floor(f*numtilings) for f in floats]
    Tiles: List[int] = []
    
    for tiling in range(numtilings):
        tilingX2: int = tiling*2
        coords: List[int] = [tiling]
        b: int = tiling
        
        for q, width in zip_longest(qfloats, wrapwidths):
            c: int = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
            
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
        
    return Tiles

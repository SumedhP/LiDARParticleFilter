from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class MapConfig:
    map_file: str # png to a map
    threshold: int # Above this value, pixels are considered occupied
    degrees_per_location: float # How many bins for degrees are made, 360 would be 1 degree accuracy

"""
This class is used to represent a map as an occupancy grid, with boolean values for each pixel.
This class also does raycasting to find the distance to the nearest occupied pixel
"""
class OccupancyGrid:
    def __init__(self, config: MapConfig):
        self.config = config
        self.degrees = 360
        self.__setupMap()
        self.__computeLookupTable()
        
    """
    Sets up the passed in map file as a boolean array
    """
    def __setupMap(self):
        self.map = Image.open(self.config.map_file)
        self.map = self.map.convert('L')
        self.map = np.array(self.map)
        self.map = self.map < self.config.threshold
    
    """
    Precomputes the lookup table for all distances to the nearest wall for all x, y, and theta
    """
    def __computeLookupTable(self) -> None:
        # Makes map of x,y,theta
        height = self.map.shape[0]
        width = self.map.shape[1]
        
        self.lt = np.zeros((height, width, self.degrees))
        
        from tqdm import tqdm
        for x in tqdm(range(height)):
            for y in range(width):
                for theta in range(self.degrees):
                    self.lt[x, y, theta] = self.computeDistanceToWall(x, y, theta)
        
        # Save the lookup table
        np.save("lookup_table.npy", self.lt)

    def isOccupied(self, x: int, y: int) -> bool:
        x = int(x)
        y = int(y)
        # If out of bounds, return occupied
        if x < 0 or x >= self.map.shape[1] or y < 0 or y >= self.map.shape[0]:
            return True
        
        return self.map[y, x]

    # This function is an implementation of the breshenham line algorithm
    def computeDistanceToWall(self, x: int, y: int, theta: float) -> float:
        # Check if the x and y are within the map
        if x < 0 or x >= self.map.shape[1] or y < 0 or y >= self.map.shape[0]:
            return 0
        
        # Find the unit vector for the direction
        dx = np.cos(theta)
        dy = np.sin(theta)
        
        # Find the distance to the nearest occupied pixel
        while not self.isOccupied(x, y):
            x += dx
            y += dy
        return np.sqrt((x - x) ** 2 + (y - y) ** 2)

if __name__ == "__main__":
    config = MapConfig("Occupancy grid example.png", 128, 360)
    og = OccupancyGrid(config)

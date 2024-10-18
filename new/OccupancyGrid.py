from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numba
from numba import prange


@dataclass
class MapConfig:
    map_file: str # png to a map
    threshold: int # Above this value, pixels are considered occupied
    lookup_table: str # File containing lookup table for distance to nearest wall, computes if not found

"""
This class is used to represent a map as an occupancy grid, with boolean values for each pixel.
This class also does raycasting to find the distance to the nearest occupied pixel
"""
class OccupancyGrid:
    def __init__(self, config: MapConfig):
        self.config = config
        self.degrees = 360
        self.__setupMap()
        # self.__displayMap()
        
        if config.lookup_table is not None:
            self.lt = np.load(config.lookup_table)
        else:
            self.__computeLookupTable()
        # self.__displayLookupTable()
        
        
    """
    Sets up the passed in map file as a boolean array
    """
    def __setupMap(self):
        self.map = Image.open(self.config.map_file)
        self.map = self.map.convert('L')
        self.map = np.array(self.map)
        self.map = self.map < self.config.threshold

    def __displayMap(self):
        print("Map shape: ", self.map.shape)
        plt.figure("Map")
        plt.imshow(self.map)
    
    """
    Precomputes the lookup table for all distances to the nearest wall for all x, y, and theta
    """
    def __computeLookupTable(self) -> None:
        # Makes map of x,y,theta
        height = self.map.shape[0] # Y-axis
        width = self.map.shape[1] # X-axis
        
        self.lt = np.zeros((width, height, self.degrees))
        print("Lt shape: ", self.lt.shape)
        
        from tqdm import tqdm
        for x in tqdm(range(width)):
            for y in range(height):
                for theta in range(self.degrees):
                    self.lt[x, y, theta] = self.__computeDistanceToWall(x, y, theta)
        
        # Save the lookup table
        output_file = self.config.map_file.split(".")[0] + "_lookup_table.npy"
        np.save(output_file, self.lt)
    
    def __displayLookupTable(self) -> None:
        plt.figure("Lookup Table")
        # Swap the x and y axis for display
        swapped = np.swapaxes(self.lt, 0, 1)
        print("Swapped shape: ", swapped.shape)
        plt.imshow(swapped[:, :, 0])
        plt.show()

    """
    Takes in a value in (x,y) value and returns if the pixel is occupied
    Interal look up happens with (y,x) since map is stored in (y,x) format
    """
    def isOccupied(self, x: int, y: int) -> bool:
        x = int(x)
        y = int(y)
        
        # If out of bounds, return occupied
        if x < 0 or x >= self.map.shape[1] or y < 0 or y >= self.map.shape[0]:
            print("Out of bounds look up at ", x, y)
            return True
        
        return self.map[y, x]

    # This function is an implementation of the breshenham line algorithm
    def __computeDistanceToWall(self, x: int, y: int, theta: float) -> float:
        # Check if the x and y are within the map
        if x < 0 or x >= self.map.shape[1] or y < 0 or y >= self.map.shape[0]:
            return 0
        
        # Find the unit vector for the direction
        dx = np.cos(theta)
        dy = np.sin(theta)
        
        # Store the starting point
        starting_x = x
        starting_y = y
        
        # Find the distance to the nearest occupied pixel
        while not self.isOccupied(x, y):
            x += dx
            y += dy
        return np.sqrt((x - starting_x) ** 2 + (y - starting_y) ** 2)
    
    """
    Returns distance to nearest wall for a given x, y, and theta
    Returns 0 if out of bounds
    Units are in pixels
    """
    def getDistance(self, x: int, y: int, theta: float) -> float:
        # Normalize the theta
        theta = theta % 360
        theta = int(theta)
        return self.lt[x, y, theta]
    
    """
    Returns distance to nearest wall for a given set of x, y, and thetas
    Returns 0 if out of bounds
    Units are in pixels
    """
    def getDistanceVectorized(self, x: np.array, y: np.array, theta: np.array) -> np.array:
        theta = theta.astype(int)
        return self.__getDistanceNumbaOptimized(self.lt, x, y, theta)
    
    @staticmethod
    @numba.njit(parallel=True)
    def __getDistanceNumbaOptimized(lt: np.array, x: np.array, y: np.array, theta: np.array) -> np.array:
        """
        This is the Numba-accelerated version of the getDistanceVectorized function.
        It takes in the lookup table and vectorized inputs for x, y, and theta.
        """
        result = np.empty(x.shape, dtype=lt.dtype)  # Create an empty array for results
        theta = theta % 360  # Ensure theta is within [0, 359] range
        
        # Loop over all x, y, theta using Numba for speed
        for i in prange(x.shape[0]):  # Use parallel execution
            result[i] = lt[x[i], y[i], theta[i]]
        
        return result

if __name__ == "__main__":
    config = MapConfig("Occupancy grid example 2.png", 128, "Occupancy grid example 2_lookup_table.npy")
    og = OccupancyGrid(config)
    values_to_test = [(14,14), (15,15), (50, 100)]
    
    for x, y in values_to_test:
        print("Pixel at ", x, y)
        print("Lookup table: ", og.getDistance(x, y, 0))

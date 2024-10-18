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
                    self.lt[x, y, theta] = self.computeDistanceToWall(x, y, theta)
        
        # Save the lookup table
        output_file = self.config.map_file.split(".")[0] + "_lookup_table.npy"
        np.save(output_file, self.lt)
    
    def __displayLookupTable(self) -> None:
        print("Lookup table shape: ", self.lt.shape)
        values_to_test = [(14,14), (15,15), (50, 100)]
        
        for x, y in values_to_test:
            print("Pixel at ", x, y)
            print("Lookup table: ", self.lt[x, y, 0])
            print("Computed: ", self.computeDistanceToWall(x, y, 0))
        
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
    def computeDistanceToWall(self, x: int, y: int, theta: float) -> float:
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
    
    def getDistance(self, x: int, y: int, theta: float) -> float:
        # Normalize the theta
        theta = theta % 360
        theta = int(theta)
        return self.lt[x, y, theta]
    
    def getDistanceVectorized(self, x: np.array, y: np.array, theta: np.array) -> np.array:
        theta = theta.astype(int)
        # return self.lt[x, y, theta]
        return self.numbaAttempt(self.lt, x, y, theta)
    
    @staticmethod
    @numba.njit(parallel=True)
    def numbaAttempt(lt: np.array, x: np.array, y: np.array, theta: np.array) -> np.array:
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

    def warmupNumba(self):
        x = np.random.randint(0, 100, 1000)
        y = np.random.randint(0, 200, 1000)
        theta = np.random.randint(0, 360, 1000)
        self.getDistanceVectorized(x, y, theta)


def runOccupancyGridTest():
        config = MapConfig("Occupancy grid example 2.png", 128, "Occupancy grid example 2_lookup_table.npy")
        og = OccupancyGrid(config)
        
        # Time how long it takes to get the distance to the nearest wall using nano seconds
        from time import process_time_ns
        start = process_time_ns()
        val = og.getDistance(14, 14, 0)
        end = process_time_ns()
        print("Time to get distance in ms: ", (end - start) / 1000000)
        
        # Now do 1000 lookups of random locations and thetas in the map. Map is 100x200
        NUM_LOOKUPS = 50000
        start = process_time_ns()
        for i in range(NUM_LOOKUPS):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 200)
            theta = np.random.randint(0, 360)
            val = og.getDistance(x, y, theta)
        end = process_time_ns()
        print("Time to get distance 1000 times in ms: ", (end - start) / 1000000)
        
        # Now do vectorized lookups of 1000 random locations and thetas in the map
        x = np.random.randint(0, 100, NUM_LOOKUPS)
        y = np.random.randint(0, 200, NUM_LOOKUPS)
        theta = np.random.randint(0, 360, NUM_LOOKUPS)
        
        #Warmups
        og.warmupNumba()
        
        from tqdm import tqdm
        start = process_time_ns()
        for i in tqdm(range(20000)):
            x = np.random.randint(0, 100, NUM_LOOKUPS)
            y = np.random.randint(0, 200, NUM_LOOKUPS)
            theta = np.random.randint(0, 360, NUM_LOOKUPS)
            val = og.getDistanceVectorized(x, y, theta)
        end = process_time_ns()
        print(f'Time to get distance {NUM_LOOKUPS} times vectorized in ms: ', (end - start) / 1000000)

if __name__ == "__main__":
    runOccupancyGridTest()

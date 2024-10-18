from OccupancyGrid import OccupancyGrid, MapConfig
import numpy as np

def runOccupancyGridTest():
        config = MapConfig("Occupancy grid example 2.png", 100, "Occupancy grid example 2_lookup_table.npy")
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
        val = og.getDistanceVectorized(x, y, theta) # Warm up call so it compiles
        
        
        from tqdm import tqdm
        start = process_time_ns()
        for i in range(1000):
            x = np.random.randint(0, 100, NUM_LOOKUPS)
            y = np.random.randint(0, 200, NUM_LOOKUPS)
            theta = np.random.randint(0, 360, NUM_LOOKUPS)
            val = og.getDistanceVectorized(x, y, theta)
        end = process_time_ns()
        print(f'Time to get distance {NUM_LOOKUPS} times vectorized in ms: ', (end - start) / 1000000)

if __name__ == "__main__":
    runOccupancyGridTest()

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
        
        start = process_time_ns()
        val = og.getDistanceVectorized(x, y, theta) # Warm up call so it compiles
        end = process_time_ns()
        print("Time to get distance vectorized warmup in ms: ", (end - start) / 1000000)
        
        from tqdm import tqdm
        NUM_ITERATIONS = 1000
        rand_x_list = np.random.randint(0, 100, NUM_LOOKUPS * NUM_ITERATIONS)
        rand_y_list = np.random.randint(0, 200, NUM_LOOKUPS * NUM_ITERATIONS)
        rand_theta_list = np.random.randint(0, 360, NUM_LOOKUPS * NUM_ITERATIONS)

        NUM_LOOPS = 100

        start = process_time_ns()
        for loop in range(NUM_LOOPS):
          for i in range(NUM_ITERATIONS):
              list_start = i * NUM_LOOKUPS
              list_end = (i + 1) * NUM_LOOKUPS
              val = og.getDistanceVectorized(rand_x_list[list_start:list_end], rand_y_list[list_start:list_end], rand_theta_list[list_start:list_end])
        end = process_time_ns()
        print("Time to get distance vectorized", NUM_ITERATIONS * NUM_LOOPS, " times for ", NUM_LOOKUPS," values in ms: ", (end - start) / 1000000 / NUM_ITERATIONS / NUM_LOOPS)
            

if __name__ == "__main__":
    runOccupancyGridTest()

from dataclasses import dataclass
# import range_libc
import numpy as np
import random

@dataclass
class LidarConfig:
    max_range_meter: float = 12.0 # D500 max range
    measurement_std_dev_m: float = 0.05 # Figure out what this should be

@dataclass
class ParticleFilterConfig:
    LidarConfig: LidarConfig
    map_file: str # This is a png
    map_px_per_meter: int # = 100 # We shold see if this should be 1px per cm or 1px per mm, affects size
    num_particles: int # = 500 
    initial_position_std_dev: float # We should measure how much play the sentry has in the square
    initial_heading_std_dev: float # Same here
    initial_x_meter: float #= 0.9 # Initial x position
    initial_y_meter: float #= 0.9 # Initial y position 
    theta_discretization: int #= 120 # What papers said to use, idk what this is
    inv_squash_factor: float #= 0.2 # What class used, idk why we do it
    movement_position_std_dev: float # Measure / guess
    movement_heading_std_dev: float # Measure / guess

class ParticleFilter:
    def __init__(self, config: ParticleFilterConfig):
        self.config = config
        self.initParticles()
        self.setupMap()

    """
    Sets up the particles with a normal distribution around the initial position
    Each partcles weight is equal
    """
    def initParticles(self):
        x = self.config.initial_x_meter * self.config.map_px_per_meter
        y = self.config.initial_y_meter * self.config.map_px_per_meter
        x_std = self.config.initial_position_std_dev * self.config.map_px_per_meter
        y_std = self.config.initial_position_std_dev * self.config.map_px_per_meter
        theta_std = self.config.initial_heading_std_dev
        N = self.config.num_particles
        
        self.weights[:] = 1 / N
        self.particles[:, 0] = np.random.normal(loc=x, scale=x_std, size=N)
        self.particles[:, 1] = np.random.normal(loc=y, scale=y_std, size=N)
        self.particles[:, 2] = np.random.normal(loc=0, scale=theta_std, size=N)

    '''
    Sets up the likelihood table for the sensor model
    Probabilty at an index (i, j) is the chance of measuring i when the true distance is j
    '''
    def precomputeSenorModel(self) -> np.ndarray:
        table_width = self.config.LidarConfig.max_range_meter + 1
        prob_table = np.zeros((table_width, table_width))
        std_dev = self.config.LidarConfig.measurement_std_dev_m
        
        def gaussian(x, mu, sig):
            return (
                1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
            )
        
        for row in range(table_width):
            for col in range(table_width):
                prob_table[row, col] = gaussian(row, col, std_dev)
        
        # Sum to a normalized probability
        prob_table = prob_table / np.sum(prob_table, axis=0)
        
        return prob_table
    
    """
    Creates the field map and setups the ray casting method
    Currently uses gpu accelerated ray casting, other methods are available
    """
    def setupMap(self):
        # map = range_libc.PyOMap(self.config.map_file)
        max_range_px = self.config.LidarConfig.max_range_meter * self.config.map_px_per_meter
        # self.range_method.set_sensor_model(self.precompute_sensor_model())
        # self.range_method = range_libc.PyRayMarchingGPU(o_map, max_range_px)
        
    def expected_pose(self) -> np.ndarray:
        """
        This was copied directly from my class code 
        
        Compute the expected state, given current particles and weights.

        Use cosine and sine averaging to more accurately compute average theta.

        To get one combined value, use the dot product of position and weight vectors
        https://en.wikipedia.org/wiki/Mean_of_circular_quantities

        Returns:
            np.array of the expected state
        """
        cosines = np.cos(self.particles[:, 2])
        sines = np.sin(self.particles[:, 2])
        theta = np.arctan2(np.dot(sines, self.weights), np.dot(cosines, self.weights))
        position = np.dot(self.particles[:, 0:2].transpose(), self.weights)

        return np.array((position[0], position[1], theta), dtype=np.float32)
        
    """
    Updates the particles based on the lidar data, recomputes the weights
    """
    def lidarUpdate(self, obs_ranges, obs_angles):
        num_rays = len(obs_ranges)
        num_particles = self.config.num_particles
        
        if self.expected_ranges is None:
            self.expected_ranges = np.zeros(num_rays * num_particles, dtype=np.float32)
            
        # # Raycasting to get expected measurements
        # self.range_method.calc_range_repeat_angles(
        #     self.particles, obs_angles, self.expected_ranges
        # )
        
        # Evaluate the sensor model
        # self.range_method.eval_sensor_model(
        #     obs_ranges, self.expected_ranges, self.weights, num_rays, num_particles
        # )

        # Squash weights to prevent too much peakiness
        np.power(self.weights, self.config.inv_squash_factor, self.weights)
        self.weights /= np.sum(self.weights)
        
    """
    Shifts all particles by the given deltas and adds some noise
    """
    def odometryUpdate(self, delta_x, delta_y, delta_theta):
        self.particles[:, 0] += delta_x
        self.particles[:, 1] += delta_y
        self.particles[:, 2] += delta_theta
        
        # Add noise
        self.particles[:, 0] += np.random.normal(0, self.config.movement_position_std_dev, self.config.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.config.movement_position_std_dev, self.config.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.config.movement_heading_std_dev, self.config.num_particles)
        
    """
    Also just copied from my class code
    This resamples the particles based on their weights
    Low Variance Resampling
    """
    def resample(self):
        N = self.config.num_particles
        r = (np.indices(dtype=float, dimensions=(N,)) / N) + random.random() / N
        self.particles[:] = self.particles[np.searchsorted(a=np.cumsum(self.weights), v=r)]
        self.weights[:] = 1 / N



import time

particles = np.zeros((500, 3))
weights = np.zeros(500)
weights[:] = 1 / 500

delta_x = 0.1
delta_y = 0.1
delta_theta = 0.1

def testOdoUpdate():
    particles[:, 0] += delta_x
    particles[:, 1] += delta_y
    particles[:, 2] += delta_theta
    
    # Add noise
    particles[:, 0] += np.random.normal(0, 0.1, 500)
    particles[:, 1] += np.random.normal(0, 0.1, 500)
    particles[:, 2] += np.random.normal(0, 0.1, 500)
    
def getExpectedPose():
    cosines = np.cos(particles[:, 2])
    sines = np.sin(particles[:, 2])
    theta = np.arctan2(np.dot(sines, weights), np.dot(cosines, weights))
    position = np.dot(particles[:, 0:2].transpose(), weights)

    return np.array((position[0], position[1], theta), dtype=np.float32)

print(getExpectedPose())

start_time = time.perf_counter()
for i in range(1000):
    testOdoUpdate()
end_time = time.perf_counter()

# print in ms
print((end_time - start_time) * 1000, " ms taken for 1000 odom updates")
# Seems to take 0.06 - 0.15ms
print(getExpectedPose())

weights = np.random.rand(500)
weights /= np.sum(weights)

def testResample():
    N = 500
    r = (np.indices(dtype=float, dimensions=(N,)) / N) + random.random() / N
    particles[:] = particles[np.searchsorted(a=np.cumsum(weights), v=r)]
    weights[:] = 1 / N

testResample()
print(getExpectedPose(), " New pose after 1 resample")

start_time = time.perf_counter()
for i in range(1000):
    testResample()
end_time = time.perf_counter()
print((end_time - start_time) * 1000, " ms taken for 1000 resamples")
print(getExpectedPose(), " New pose after 1000 resamples")


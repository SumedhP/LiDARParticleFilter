from dataclasses import dataclass
import numpy as np
import numba
from numba import prange

@dataclass
class SensorModelConfig:
    measurement_std_dev_m: float
    max_range_meter: float = 12.0 # D500 max range
    inv_squash_factor: float = 1.0 # Inverse squash factor for the sensor model

"""
This class is used to represent the LiDAR, giving the probabilty of a measurement given the true distance.
The probabilty is stored in a 2D array, where the row is the observed distance and the column is the expected distance.
The value is the probability of the observed distance given the expected distance.

This is then normalized such that all the values in a column sum to 1, 
i.e. given the true distance, which of the rows is most likely.
"""
class SensorModel:
    def __init__(self, config: SensorModelConfig):
        self.config = config
        self.__precomputeSensorModel()
        
    def __precomputeSensorModel(self) -> None:
        table_width = self.config.max_range_meter + 1
        self.prob_table = np.zeros((table_width, table_width))
        std_dev = self.config.measurement_std_dev_m
        
        for i in range(table_width):
            for j in range(table_width):
                self.prob_table[i, j] = self.__guassian(i, j, std_dev)
        
        self.prob_table /= np.sum(self.prob_table, axis=0) # Normalize the table
        
    def __guassian(self, x, mu, sig):
        return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )
    
    def getProbabilityTable(self) -> np.ndarray:
        return self.prob_table
    
    """
    Expects a nx1 array of observed distances and a nx1 array of expected distances
    Meant for singular particle evaluation
    """
    def getProbability(self, obsDistances, expectedDistances) -> float:
        prob = 1.0
        for i in range(len(obsDistances)):
            prob *= self.prob_table[obsDistances[i], expectedDistances[i]]
        return prob
    
    """
    Expects a n x d array of observed distances and a n x d array of expected distances
    n is the number of particles and d is the number of obeservations
    """
    def getProbabilityVectorized(self, obsDistances, expectedDistances) -> np.ndarray:
        return self.__getProbabilityNumbaOptimized(obsDistances, expectedDistances, self.prob_table, self.config.inv_squash_factor)
    
    @staticmethod
    @numba.njit(parallel=True)
    def __getProbabilityNumbaOptimized(obsDistances, expectedDistances, prob_table, inv_squash_factor) -> np.ndarray:
        """
        Generated with the help of AI, this is a numba optimized version of the getProbabilityVectorized method 
        """
        # Initialize probability array with ones
        n_particles = obsDistances.shape[0]
        n_observations = obsDistances.shape[1]
        
        prob = np.ones(n_particles)
        
        # Loop through all observations in parallel
        for i in numba.prange(n_observations):  # Parallelized outer loop (over observations)
            for j in range(n_particles):  # Iterate over particles
                obs_dist = obsDistances[j, i]
                exp_dist = expectedDistances[j, i]
                prob[j] *= prob_table[obs_dist, exp_dist]
        
        # Squash the probabilities to avoid peakiness
        prob = np.power(prob, inv_squash_factor) 
        
        # Normalize the probabilities
        prob /= np.sum(prob)
        return prob
        

if __name__ == "__main__":
    config = SensorModelConfig(3, 5)
    sm = SensorModel(config)
    pt = sm.getProbabilityTable()
    # Print in a nice matrix format, force to print in decimal and not scientific notation
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    for i in range(pt.shape[0]):
        print(pt[i, :])

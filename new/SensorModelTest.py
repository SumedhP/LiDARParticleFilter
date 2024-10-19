# Run individual lookups and vectorized lookups

from SensorModel import SensorModel, SensorModelConfig
import numpy as np

def runSensorModelTest():
    config = SensorModelConfig(5, 10, 0.2)
    sm = SensorModel(config)
    
    # Print the table
    pt = sm.getProbabilityTable()
    # Print in a nice matrix format, force to print in decimal and not scientific notation
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    for i in range(pt.shape[0]):
        print(pt[i, :])

    NUM_PARTICLES = 2000
    NUM_RAYS = 24
        
    # Run a single particle evaluation
    obsDistances = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expectedDistances = np.array([2, 2, 1, 4, 7, 6, 7, 3, 9, 9])
    
    prob = sm.getProbability(obsDistances, expectedDistances)
    # print with 10 decimal places
    print("Probability for single particle: ", "{0:0.20f}".format(prob))
    
    # Run a vectorized particle evaluation
    obsDistances = np.random.randint(0, 10, (NUM_PARTICLES, NUM_RAYS))
    expectedDistances = np.random.randint(0, 10, (NUM_PARTICLES, NUM_RAYS))
    prob = sm.getProbabilityVectorized(obsDistances, expectedDistances)

    # Now repeat each evaluation method
    AMOUNT_OF_EVALS = 1000
    for i in range(AMOUNT_OF_EVALS):
        obsDistances = np.random.randint(0, 10, (NUM_PARTICLES, NUM_RAYS))
        expectedDistances = np.random.randint(0, 10, (NUM_PARTICLES, NUM_RAYS))
        
        # Single particle evaluation
        prob = sm.getProbability(obsDistances[0, :], expectedDistances[0, :])
        
        # Vectorized particle evaluation
        prob = sm.getProbabilityVectorized(obsDistances, expectedDistances)

    print("The most likely particle has a probability of: ", np.max(prob))

if __name__ == "__main__":
    runSensorModelTest()

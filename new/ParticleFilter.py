from dataclasses import dataclass
import numpy as np
import OccupancyGrid
import SensorModel


@dataclass
class ParticleFilterConfig:
    num_particles: int
    map_px_per_meter: int
    initial_position_std_dev: float
    initial_heading_std_dev: float
    movement_position_std_dev: float
    movement_heading_std_dev: float
    initial_x_meter: float
    initial_y_meter: float


class ParticleFilter:
    def __init__(
        self,
        config: ParticleFilterConfig,
        sensorModelConfig: SensorModel.SensorModelConfig,
        mapConfig: OccupancyGrid.MapConfig,
    ):
        self.config = config
        self.sensorModel = SensorModel.SensorModel(sensorModelConfig)
        self.map = OccupancyGrid.OccupancyGrid(mapConfig)
        self.initParticles()

    def initParticles(self):
        N = self.config.num_particles

        self.particles = np.zeros((N, 3))
        self.particles[:, 0] = np.random.normal(
            loc=self.config.initial_x_meter,
            scale=self.config.initial_position_std_dev,
            size=N,
        )
        self.particles[:, 1] = np.random.normal(
            loc=self.config.initial_y_meter,
            scale=self.config.initial_position_std_dev,
            size=N,
        )
        self.particles[:, 2] = np.random.normal(
            loc=0, scale=self.config.initial_heading_std_dev, size=N
        )
        self.weights = np.ones(N) / N

    def getExpectedPosition(self):
        return np.average(self.particles, axis=0, weights=self.weights)
        """
        Code from the class, I don't know if it's any more accurate:
        cosines = np.cos(self.particles[:, 2])
        sines = np.sin(self.particles[:, 2])
        theta = np.arctan2(np.dot(sines, self.weights), np.dot(cosines, self.weights))
        position = np.dot(self.particles[:, 0:2].transpose(), self.weights)

        return np.array((position[0], position[1], theta), dtype=np.float32)
        """
    
    def getParticles(self):
        return self.particles

    def __resample(self):
        """
        Low Variance Resampling.
        Directly copied from the class code.
        """
        N = self.config.num_particles
        r = (np.indices(dtype=float, dimensions=(N,)) / N) + np.random.random() / N
        self.particles[:] = self.particles[np.searchsorted(a=np.cumsum(self.weights), v=r)]
        self.weights[:] = 1 / N
    
    def odometryUpdateCallback(self, dx_meter, dy_meter, dtheta_deg):
        self.particles[:, 0] += dx_meter
        self.particles[:, 1] += dy_meter
        self.particles[:, 2] += dtheta_deg
        
        # Add noise
        self.particles[:, 0] += np.random.normal(0, self.config.movement_position_std_dev, self.config.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.config.movement_position_std_dev, self.config.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.config.movement_heading_std_dev, self.config.num_particles)
        
        # Make sure all particles are within the map
        mapDim = self.map.getMapDimensions()
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, mapDim[0])
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, mapDim[1])

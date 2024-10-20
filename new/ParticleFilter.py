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

    def getPosition(self):
        return np.average(self.particles, axis=0, weights=self.weights)

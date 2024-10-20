import cProfile
import pstats
from OccupancyGridTest import runOccupancyGridTest
from SensorModelTest import runSensorModelTest

cProfile.run('runSensorModelTest()', 'profiler_output')

def f8_alt(x):
    return "%1.9f" % x
  
pstats.f8 = f8_alt

p = pstats.Stats('profiler_output')
# Sort by cum time
p.strip_dirs().sort_stats('time').print_stats("OccupancyGrid|SensorModel")

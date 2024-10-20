## Let's pseudo code the Particle Filter algorithm

I make some amount of particles, N, and give them a gassuain distribution of positions and headings.

### Accesible functions of this class:
- Expected position
- Particle locations (to display on Prism)
- Reintialize around a point (For ease of development?)

### Algorithm functions:
- odometry update -> Direct shift of all particles by given meter and heading deltas
- lidar update -> Get scan, get expected positions for all particles at those positions, and evalauate
(In the future have it handle lidar offsets)
- Resample -> This should happen internally on LiDAR updates

## GLOBAL UNITS WILL BE:
Meters
Degrees

# Interactions then for each method:
- Odometry update:
Takes in a dx, dy, dtheta
Shift all particles
Add in noise
Make sure the particles are in frame

- Lidar update:
Take in a array of observed angles and the various angles at which they happened (Len: D)
We then send this off to the Map class and ask back for a N x D set of ranges
We then send the N x D set of expected Ranges and then a D set of observed ranges and then compare them, giving back a N x 1 set of weights
Either we or the sensor model normalizes these and accounts for inv_squash_factor

## As such the following methods need to occur:
Input values are in meters and degrees
output values are in meters
Make sure particle locations and map locations are in the same frame 
(x, y starts from bottom right and increases to top left,
x is up down, y is left right, and theta 0 is up and increases counter clockwise)

### Get Expected Distances ( particle positions N x 3, observed angles D) -> N X D:
output = N x D
for each particle:
  for each angle:
    Convert particle location from meters to px value
    heading is particle heading + angle heading
    output[particle][range] = lut[x][y][theta] # Convert this from px to meters
return output 

### Evaluate Sensor Model ( Observed ranges N x D, actual ranges D) -> N x 1:
output = N x 1 of 1s
Here all distances are probably casted down into ints? Doesn't give us much "resolution" if we have a narrow std dev
for each particle:
  for each range:
    output[particle] *= probability[obs][expected]
return output



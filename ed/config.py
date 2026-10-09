import numpy as np

DTYPE = np.complex128

# Lattice
CELL = 'honeycomb'
CLUSTER = 'A'
N_MAX = np.array([2, 3])

# Occupation
FILLING = 0.5   # half filling is 0.5
SZ = 0.0

# Hoppings and twisted boundary flux
T1 = 1.0
T2 = 0.2
PHI = 0.5 # in units of pi

# ED solver parameters
TOL = 0

# Flux grid per dimension
N_FLUX = 4

# Disorder parameters
N_SAMPLES = 1
W0 = 0.0
W1 = 0.0
W2 = 0.0
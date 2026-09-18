import numpy as np

NUM_SITES = 12
NUM_ELECTRONS = 12   # half filling at 12 electrons
N_UP = NUM_ELECTRONS // 2
N_DN = NUM_ELECTRONS - N_UP

# Hoppings and twisted boundary flux
T1 = 1.0
T2 = 0.2
PHI = 0.5 # in units of pi

# ED solver parameters
K = 5
TOL = 0

# Flux grid per dimension
N_FLUX = 4

# Disorder parameters
N_SAMPLES = 1
W0 = 0.0
W1 = 0.0
W2 = 0.0

DTYPE = np.complex128
VERBOSE = False
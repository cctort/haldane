import numpy as np

NUM_SITES = 12
NUM_ELECTRONS = 3   # half filling at 12 electrons
N_UP = NUM_ELECTRONS // 2
N_DN = NUM_ELECTRONS - N_UP

# Hoppings and twisted boundary flux
T1 = 1.0
T2 = 0.2
PHI = np.pi / 2

# ED tolerance
TOL = 1e-11

# Flux grid per dimension
N_GRID = 4

DTYPE = np.complex128
VERBOSE = False

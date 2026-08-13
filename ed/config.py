"""
Configuration and physical constants for Haldane-Hubbard model.
"""

import numpy as np

# Honeycomb lattice parameters
# 12A cluster: 12 sites, 24 orbitals (2 spin states)
NUM_SITES = 12
SPINS = 2
NUM_ORBITALS = NUM_SITES * SPINS
HILBERT_DIM = 2 ** NUM_ORBITALS

# 12 electrons is half-filling
NUM_ELECTRONS = 12

T1 = 1.0  # NN hopping amplitude
T2 = 0.2  # NNN hopping amplitude
PHI = np.pi / 2  # Chirality phase

DELTA_DEFAULT = 0.0  # Staggered potential
U_DEFAULT = 0.0  # Hubbard interaction
V_DEFAULT = 0.0  # Nearest-neighbor density-density

# Phase diagram scanning ranges
DELTA_RANGE = np.linspace(0, 4, 8)
U_RANGE = np.linspace(0, 13, 8)
V_RANGE = np.linspace(0, 4, 8)

FLUX_NPOINTS = 16  # Will compute on NxN grid of flux points
CHERN_INTEGRATION_POINTS = 2  # {ρ(0), ρ(π)}

KRYLOV_K = 1  # Just want ground state
WHICH = 'SA'  # Smallest (lowest) eigenvalue
TOL = 1e-12  # ED convergence tolerance
MAXITER = 10000  # ED maximum iterations

DTYPE = np.complex128
DTYPE_REAL = np.float64

DATA_DIR = './data'
RESULTS_FILE = 'phase_diagram.h5'

VERBOSE = False

def print_config():
    """Print configuration summary."""
    print("=" * 70)
    print("HALDANE-HUBBARD MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Lattice: Honeycomb 12A cluster")
    print(f"  Sites: {NUM_SITES}")
    print(f"  Orbitals: {NUM_ORBITALS} (with spin)")
    print(f"  Hilbert space: 2^{NUM_ORBITALS} = {HILBERT_DIM:.2e}")
    print(f"  Electrons (half-filling): {NUM_ELECTRONS}")
    print(f"\nHopping parameters:")
    print(f"  t1 = {T1}")
    print(f"  t2 = {T2} (= {T2/T1}t1)")
    print(f"  φ = {PHI/np.pi}π")
    print(f"\nDefaults:")
    print(f"  Δ = {DELTA_DEFAULT}, U = {U_DEFAULT}, V = {V_DEFAULT}")
    print("=" * 70)

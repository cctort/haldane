"""
Configuration and physical constants for Haldane-Hubbard model.
"""

import numpy as np

# ============================================================================
# LATTICE PARAMETERS
# ============================================================================

# Honeycomb lattice parameters
# Using 12A cluster: 12 sites, 24 orbitals (2 spin states)
NUM_SITES = 12
SPINS = 2
NUM_ORBITALS = NUM_SITES * SPINS  # 24
HILBERT_DIM = 2 ** NUM_ORBITALS  # ~16 million

# Half-filling: 1 electron per site => 12 electrons across 24 spin-orbitals
# (NOT 6 -- 6 was a bug: it filled only a quarter of the spin-orbitals,
# i.e. on average half of the lower band per spin, which is metallic rather
# than the gapped/topological filling needed to reproduce the phase diagram.)
NUM_ELECTRONS = 12

# ============================================================================
# HOPPING PARAMETERS
# ============================================================================

# Set t as energy scale (t = 1)
T1 = 1.0  # First-neighbor hopping amplitude
T2 = 0.2  # Second-neighbor hopping amplitude

# Chirality phase (maximizes topological region)
PHI = np.pi / 2

# ============================================================================
# INTERACTION PARAMETERS
# ============================================================================

# Will be varied in phase diagram
# Default values for testing
DELTA_DEFAULT = 0.0  # Staggered potential
U_DEFAULT = 0.0  # Hubbard interaction
V_DEFAULT = 0.0  # Nearest-neighbor density-density

# Phase diagram scanning ranges
DELTA_RANGE = np.linspace(0, 4.1, 2)
U_RANGE = np.linspace(0, 10.5, 2)
V_RANGE = np.linspace(0, 4.1, 2)

# ============================================================================
# CHERN NUMBER CALCULATION
# ============================================================================

# Flux grid for Berry curvature integration
# Using high-symmetry points: ρ(0) and ρ(π)
FLUX_NPOINTS = 16  # Will compute on NxN grid of flux points
CHERN_INTEGRATION_POINTS = 2  # {ρ(0), ρ(π)}

# ============================================================================
# ED SOLVER PARAMETERS
# ============================================================================

# Krylov space dimension (k-value for eigsh)
KRYLOV_K = 1  # Just want ground state

# Eigenvalue target
WHICH = 'SA'  # Smallest (lowest) eigenvalue

# Convergence tolerance
TOL = 1e-12

# Maximum iterations
MAXITER = 10000

# ============================================================================
# NUMERICAL PRECISION
# ============================================================================

DTYPE = np.complex128  # Complex arithmetic
DTYPE_REAL = np.float64

# ============================================================================
# I/O AND CACHING
# ============================================================================

DATA_DIR = './data'
RESULTS_FILE = 'phase_diagram.h5'
CHECKPOINT_INTERVAL = 5  # Save every N points

# ============================================================================
# DEBUGGING
# ============================================================================

VERBOSE = True
PROFILE = False  # Enable timing profiling

def print_config():
    """Print configuration summary."""
    print("=" * 70)
    print("HALDANE-HUBBARD MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Lattice: Honeycomb 12A cluster")
    print(f"  Sites: {NUM_SITES}")
    print(f"  Orbitals: {NUM_ORBITALS} (with spin)")
    print(f"  Hilbert space: 2^{NUM_ORBITALS} ≈ {HILBERT_DIM:.2e}")
    print(f"  Electrons (half-filling): {NUM_ELECTRONS}")
    print(f"\nHopping parameters:")
    print(f"  t1 = {T1}")
    print(f"  t2 = {T2} (= {T2/T1}·t1)")
    print(f"  φ = {PHI/np.pi}π")
    print(f"\nDefaults:")
    print(f"  Δ = {DELTA_DEFAULT}, U = {U_DEFAULT}, V = {V_DEFAULT}")
    print("=" * 70)

"""
Haldane-Hubbard Model Exact Diagonalization Code.

Python implementation for reproducing results from:
"Emergence of an antiferromagnetic topological Anderson insulator in the 
interacting Haldane model" - Uría-Álvarez & Valentí (2026)

Main modules:
- config: Configuration and parameters
- basis: Fixed particle number Fock basis
- lattice: Honeycomb lattice structure
- hamiltonian: Many-body Hamiltonian construction
- ed_solver: Exact diagonalization solver
- observables: Observable calculations (CDW, SDW, Chern number)

Future: C++ kernels for Hamiltonian building and diagonalization
"""

__version__ = "0.1.0"

from . import config
from . import basis
from . import lattice
from . import hamiltonian
from . import ed_solver
from . import observables

__all__ = [
    'config',
    'basis',
    'lattice', 
    'hamiltonian',
    'ed_solver',
    'observables',
]

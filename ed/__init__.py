"""
Haldane-Hubbard Model Exact Diagonalization Code.

Reproduces "Emergence of an antiferromagnetic topological Anderson
insulator in the interacting Haldane model" (Uria-Alvarez & Valenti,
2026). ED method follows the approach in Jafari, "Introduction to
Hubbard Model and Exact Diagonalization" (arXiv:0807.4878).

Modules
-------
config       physical and numerical parameters
lattice      12A honeycomb cluster, nn/nnn bonds
basis        per-spin fixed-N Fock basis
fast_ops     bit-level hopping matrix elements
hamiltonian  spin-factorized, matrix-free Hamiltonian
ed_solver    ground state via Lanczos
observables  CDW, SDW, density matrix, Chern number
"""
from . import config
from . import lattice
from . import basis
from . import hamiltonian
from . import ed_solver
from . import observables

__all__ = [
    'config', 'lattice', 'basis', 'hamiltonian', 'ed_solver', 'observables',
]

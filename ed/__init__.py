"""
Haldane-Hubbard Model Exact Diagonalization Code.

Modules
-------
config       physical and numerical parameters
lattice      12A honeycomb cluster, nn/nnn bonds
basis        per-spin fixed-N Fock basis
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
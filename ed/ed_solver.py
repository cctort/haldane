from .config import TOL, K
from .hamiltonian import Hamiltonian
from scipy.sparse.linalg import eigsh

class EDSolver:
    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian

    def ground_state(self, delta, U, V, flux_x=0.0, flux_y=0.0, v0=None, gap=False):
        """Calculates lowest energy E_gs and ground state psi_gs of H(delta, U, V, flux)"""
        H = self.hamiltonian.build(delta, U, V, flux_x, flux_y)

        if gap:
            vals, vecs = eigsh(H, k=2, which='SA', tol=TOL, v0=v0, ncv=100)
            gap = abs(vals[1] - vals[0])
            return vals[0], vecs[:, 0], gap
        else:
            vals, vecs = eigsh(H, k=1, which='SA', tol=TOL, v0=v0, ncv=100)
            return vals[0], vecs[:, 0]
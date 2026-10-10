from .config import Config
from .hamiltonian import Hamiltonian
from scipy.sparse.linalg import eigsh

class EDSolver:
    def __init__(self, cfg: Config, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian
        self.tol = cfg.tol

    def ground_state(self, delta, U, V, flux_x=0.0, flux_y=0.0, v0=None, gap=False, site_dis=None, bond_dis=None):
        """Calculates lowest energy E_gs and ground state psi_gs of H(delta, U, V, flux)"""
        H = self.hamiltonian.build(delta, U, V, flux_x, flux_y, site_dis, bond_dis)

        if gap:
            vals, vecs = eigsh(H, k=2, which='SA', tol=self.tol, v0=v0)
            gap = abs(vals[1] - vals[0])
            return vals[0], vecs[:, 0], gap
        else:
            vals, vecs = eigsh(H, k=1, which='SA', tol=self.tol, v0=v0)
            return vals[0], vecs[:, 0]
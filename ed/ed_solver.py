from .config import TOL, T1, T2, PHI
from .hamiltonian import Hamiltonian
from scipy.sparse.linalg import eigsh

class EDSolver:
    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian

    def ground_state(self, delta, U, V, t1=T1, t2=T2, phi=PHI, flux_x=0.0, flux_y=0.0, v0=None, gap=False, site_dis=None, bond_dis=None, tol=TOL):
        """Calculates lowest energy E_gs and ground state psi_gs of H(delta, U, V, flux)"""
        H = self.hamiltonian.build(delta, U, V, t1, t2, phi, flux_x, flux_y, site_dis, bond_dis)

        if gap:
            vals, vecs = eigsh(H, k=2, which='SA', tol=tol, v0=v0)
            gap = abs(vals[1] - vals[0])
            return vals[0], vecs[:, 0], gap
        else:
            vals, vecs = eigsh(H, k=1, which='SA', tol=tol, v0=v0)
            return vals[0], vecs[:, 0]
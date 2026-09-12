import numpy as np
from .config import TOL
from .hamiltonian import Hamiltonian
from scipy.sparse.linalg import eigsh

class EDSolver:
    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian

    def ground_state(self, delta, U, V, flux_x=0.0, flux_y=0.0, v0=None, gap=False):
        """Calculates lowest energy E_gs and ground state psi_gs of H(delta, U, V, flux)"""
        H = self.hamiltonian.build(delta, U, V, flux_x, flux_y)

        if gap:
            vals, vecs = eigsh(H, k=2, which='SA', tol=TOL, v0=v0)
            gap = abs(vals[1] - vals[0])
            return vals[0], vecs[:, 0], gap
        else:
            vals, vecs = eigsh(H, k=1, which='SA', tol=TOL, v0=v0)
            return vals[0], vecs[:, 0]

    def flux_sweep(self, delta, U, V, flux_points):
        """
        Calculates lowest energy E_gs and ground state psi_gs of H(delta, U, V, flux) 
        for a list of flux points (flux_x, flux_y). 
        
        Returns an array of shape (len(flux_points), H.shape[0])
        """
        states, v0 = [], None
        for fx, fy in flux_points:
            _, psi = self.ground_state(delta, U, V, fx, fy, v0)
            states.append(psi)
            v0 = psi
        return np.array(states).T
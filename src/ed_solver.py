"""
Exact diagonalization solver.

Uses scipy.sparse.linalg.eigsh for efficient computation of ground state
via Krylov subspace methods (Lanczos algorithm).
"""

import numpy as np
from scipy.sparse import linalg as sp_linalg
from scipy import sparse
from typing import Tuple, Optional
from .config import (
    KRYLOV_K, WHICH, TOL, MAXITER, VERBOSE
)
from .hamiltonian import HaldaneHubbardHamiltonian


class ExactDiagonalizationSolver:
    """
    Solve the interacting many-body Hamiltonian via exact diagonalization.
    
    Uses Krylov subspace iterative methods for sparse eigenvalue problems.
    Finds ground state and computes expectation values of observables.
    """
    
    def __init__(self, hamiltonian: HaldaneHubbardHamiltonian):
        """
        Initialize ED solver.
        """
        self.hamiltonian = hamiltonian
        self.basis = hamiltonian.basis
        
        # Ground state cache
        self._cached_gs = None
        self._cached_gs_energy = None
        self._cached_params = None
        
        # Timing statistics
        self._timings = {}
    
    def solve(self, delta: float, U: float, V: float, flux_x: float = 0.0, flux_y: float = 0.0,
              v0: Optional[np.ndarray] = None, use_cache: bool = True
        ) -> Tuple[float, np.ndarray]:
        """
        Solve for ground state of the Hamiltonian.
        
        Args:
            delta, U, V: Model parameters
            flux_x, flux_y: Twisted boundary condition parameters
            v0: Initial guess vector for Lanczos iteration (warm start)
            use_cache: Whether to use cached results
            
        Returns:
            (E_gs, |ψ_gs⟩): Ground state energy and wavefunction
        """
        # Check cache
        params = (delta, U, V, flux_x, flux_y)
        if use_cache and self._cached_params == params and self._cached_gs is not None:
            if VERBOSE:
                print(f"  Using cached ground state for Δ={delta:.3f}, U={U:.3f}, V={V:.3f}")
            return self._cached_gs_energy, self._cached_gs
        
        if VERBOSE:
            print(f"  Solving for Δ={delta:.3f}, U={U:.3f}, V={V:.3f}", end="... ")
        
        # Build Hamiltonian
        import time
        t0 = time.time()
        H = self.hamiltonian.build_full_hamiltonian(delta, U, V, flux_x, flux_y)
        t_build = time.time() - t0
        self._timings['hamiltonian_build'] = t_build
        
        # Solve for ground state using sparse eigenvalue solver
        t0 = time.time()
        try:
            eigenvalues, eigenvectors = sp_linalg.eigsh(
                H,
                k=KRYLOV_K,
                which=WHICH,
                tol=TOL,
                maxiter=MAXITER,
                return_eigenvectors=True,
                v0=v0  # Pass warm start vector to ARPACK
            )
            t_solve = time.time() - t0
            self._timings['eigsh'] = t_solve
            
            E_gs = eigenvalues[0]
            psi_gs = eigenvectors[:, 0]
            
        except sp_linalg.ArpackNoConvergence as e:
            print(f"WARNING: Eigenvalue solver did not converge!")
            print(f"  Parameters: Δ={delta}, U={U}, V={V}")
            print(f"  Error: {e}")
            raise
        
        # Cache result
        self._cached_gs = psi_gs
        self._cached_gs_energy = E_gs
        self._cached_params = params
        
        if VERBOSE:
            print(f"E_gs = {E_gs:.6f} (build: {t_build:.3f}s, solve: {t_solve:.3f}s)")
        
        return E_gs, psi_gs
    
    def solve_flux_sector(self, delta: float, U: float, V: float, flux_points: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for ground state at multiple flux insertions.
        
        Used for calculating Chern number via Berry curvature.
        
        Args:
            delta, U, V: Model parameters
            flux_points: Array of (flux_x, flux_y) tuples. If None, uses default grid.
            
        Returns:
            (flux_angles, ground_states): Array of flux points and corresponding ground states
        """
        if flux_points is None:
            # Default: high-symmetry points {0, π}
            flux_points = np.array([
                (0.0, 0.0),
                (np.pi, 0.0),
                (0.0, np.pi),
                (np.pi, np.pi),
            ])
        
        n_points = len(flux_points)
        fock_dim = len(self.basis.basis_states)
        
        ground_states = np.zeros((fock_dim, n_points), dtype=np.complex128)
        
        v_warm = None
        for i, (fx, fy) in enumerate(flux_points):
            _, psi = self.solve(delta, U, V, flux_x=fx, flux_y=fy, v0=v_warm, use_cache=False)
            ground_states[:, i] = psi
            v_warm = psi  # Warm start next step with previous state
        
        return flux_points, ground_states
    
    def clear_cache(self):
        """
        Clear ground state cache.
        """
        self._cached_gs = None
        self._cached_gs_energy = None
        self._cached_params = None
    
    def get_timings(self) -> dict:
        """
        Get timing statistics.
        """
        return self._timings.copy()
    
    @staticmethod
    def compute_energy_density(
        psi: np.ndarray,
        observable: sparse.csr_matrix
    ) -> float:
        """
        Compute expectation value of an observable.
        
        <ψ|O|ψ>
        
        Args:
            psi: Ground state wavefunction
            observable: Sparse matrix representation of observable
            
        Returns:
            Expectation value
        """
        # Compute O|ψ⟩
        psi_transformed = observable @ psi
        
        # Compute ⟨ψ|O|ψ⟩ = ⟨ψ| (O|ψ⟩)
        expectation = np.vdot(psi, psi_transformed)
        
        return np.real(expectation)


class EDSolverFactory:
    """Factory for creating and caching ED solver instances."""
    
    def __init__(self):
        self._solver = None
    
    def get_solver(self, hamiltonian: HaldaneHubbardHamiltonian) -> ExactDiagonalizationSolver:
        """Get or create solver instance."""
        if self._solver is None:
            self._solver = ExactDiagonalizationSolver(hamiltonian)
        return self._solver
    
    def reset(self):
        """Reset solver instance."""
        if self._solver is not None:
            self._solver.clear_cache()
        self._solver = None


# Global factory
_FACTORY = EDSolverFactory()


def get_ed_solver(hamiltonian: HaldaneHubbardHamiltonian) -> ExactDiagonalizationSolver:
    """Get global ED solver instance."""
    return _FACTORY.get_solver(hamiltonian)
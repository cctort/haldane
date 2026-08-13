"""
Observable module: Calculates CDW, squared SDW correlation,
and spinful Fukui-Hatsugai-Suzuki Chern Numbers.

Rewritten for the Sz-factorized basis (basis.py): a ground state psi is a
flat vector of length fock_dim = dim_dn * dim_up, representing an implicit
(dim_dn, dim_up) array M via reshape. Since the model never mixes spin, this
structure lets every observable below be computed from small (NUM_SITES- or
dim_up/dim_dn-scale) precomputed pieces plus a couple of full-size but fully
vectorized numpy reductions/matmuls -- never a Python loop over the combined
basis.
"""

import warnings
import numpy as np
from .config import NUM_SITES, SPINS, NUM_ORBITALS, N_UP, N_DN
from .basis import SzSectorBasis
from .lattice import HoneycombLattice
from .ed_solver import ExactDiagonalizationSolver
from .fast_ops import build_orbital_pair_template, build_occupation_masks


class ObservableCalculator:
    """
    Computes physical order parameters and topological invariants.
    """

    def __init__(self, basis: SzSectorBasis, lattice: HoneycombLattice, solver: ExactDiagonalizationSolver):
        self.basis = basis
        self.lattice = lattice
        self.solver = solver
        self.dim_up = basis.dim_up
        self.dim_dn = basis.dim_dn
        self.fock_dim = basis.fock_dim

        same_sector = basis.dn is basis.up

        # Per-site occupation masks for each spin sub-basis (NUM_SITES, dim).
        self._occ_up = build_occupation_masks(basis.up.states_np, NUM_SITES).astype(np.float64)
        self._occ_dn = (self._occ_up if same_sector
                         else build_occupation_masks(basis.dn.states_np, NUM_SITES).astype(np.float64))

        self._sub_sign = np.array(
            [1.0 if lattice.site_to_sublattice[s] == 0 else -1.0 for s in range(NUM_SITES)]
        )

        # Lazy per-(orb_i, orb_j) connectivity template caches, one per spin
        # sub-basis, used by compute_single_particle_density_matrix.
        self._pair_template_cache_up = {}
        self._pair_template_cache_dn = {}

    def _get_pair_template_up(self, orb_i: int, orb_j: int):
        key = (orb_i, orb_j)
        t = self._pair_template_cache_up.get(key)
        if t is None:
            t = build_orbital_pair_template(self.basis.up.states_np, orb_i, orb_j)
            self._pair_template_cache_up[key] = t
        return t

    def _get_pair_template_dn(self, orb_i: int, orb_j: int):
        key = (orb_i, orb_j)
        t = self._pair_template_cache_dn.get(key)
        if t is None:
            t = build_orbital_pair_template(self.basis.dn.states_np, orb_i, orb_j)
            self._pair_template_cache_dn[key] = t
        return t

    @staticmethod
    def _check_normalized(psi_gs: np.ndarray, tol: float = 1e-8, label: str = "psi_gs") -> None:
        """
        Sanity check that the ground state vector is normalized before using |psi|^2 as a
        probability distribution.
        """
        norm_sq = float(np.sum(np.abs(psi_gs) ** 2))
        if abs(norm_sq - 1.0) > tol:
            warnings.warn(
                f"{label} is not normalized (sum|psi|^2 = {norm_sq:.10f}, "
                f"deviation = {norm_sq - 1.0:.2e}). Renormalizing for this calculation, "
                f"but you should check why the solver's eigenvector drifted.",
                RuntimeWarning,
            )

    def _prob_grid(self, psi_gs: np.ndarray) -> np.ndarray:
        """|psi|^2 reshaped to (dim_dn, dim_up), renormalized defensively."""
        M = np.asarray(psi_gs).reshape(self.dim_dn, self.dim_up)
        P = np.abs(M) ** 2
        norm = P.sum()
        if norm > 0:
            P = P / norm
        return P

    def _staggered_vectors(self):
        """(stag_dn, stag_up): sum_i sign_i * n_i per single-spin basis state."""
        stag_up = self._sub_sign @ self._occ_up
        stag_dn = stag_up if self._occ_dn is self._occ_up else self._sub_sign @ self._occ_dn
        return stag_dn, stag_up

    def compute_cdw(self, psi_gs: np.ndarray) -> float:
        """
        Staggered CDW structure factor S_CDW = (1/N) sum_ij xi_i xi_j <n_i n_j>.

        Vectorized: sum_i xi_i n_i = stag_dn[dn] + stag_up[up] on the (dn, up)
        grid (since n_i = n_i_dn + n_i_up splits cleanly by sub-basis), so
        sum_ij xi_i xi_j n_i n_j = (sum_i xi_i n_i)^2 -- a single reduction,
        no double loop over sites needed.
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_cdw)")
        P = self._prob_grid(psi_gs)
        stag_dn, stag_up = self._staggered_vectors()
        D = stag_dn[:, None] + stag_up[None, :]
        return float(np.sum(D * D * P) / NUM_SITES)

    def compute_sdw_squared(self, psi_gs: np.ndarray) -> float:
        """
        Staggered SDW structure factor
            S_SDW = (1/N) sum_ij xi_i xi_j <(n_i_up - n_i_dn)(n_j_up - n_j_dn)>.

        Vectorized the same way as compute_cdw: sum_i xi_i (n_i_up - n_i_dn)
        = stag_up[up] - stag_dn[dn] on the (dn, up) grid, so the double sum
        reduces to a single squared-and-summed reduction.
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_sdw_squared)")
        P = self._prob_grid(psi_gs)
        stag_dn, stag_up = self._staggered_vectors()
        Dz = stag_up[None, :] - stag_dn[:, None]
        return float(np.sum(Dz * Dz * P) / NUM_SITES)

    def compute_single_particle_density_matrix(self, psi_gs: np.ndarray) -> np.ndarray:
        """
        Compute single-particle density matrix rho_ij = <c_i^dagger c_j>,
        indexed as orb = site*SPINS + spin (spin 0 = up, 1 = down), matching
        the original flat-basis convention.

        Since the Hamiltonian never mixes spin, rho is block-diagonal in
        spin: the up-up and down-down blocks are computed independently, and
        the up-down cross block is exactly zero (not just numerically small
        -- there is no term in H that could generate it).

        Each spin block is obtained from a Gram ("partial trace") matrix of
        the reshaped wavefunction M (dim_dn, dim_up):

            G  = M^H @ M     (dim_up, dim_up):  G[a,b] = sum_dn conj(M[dn,a]) M[dn,b]
            H2 = M @ M^H     (dim_dn, dim_dn):  H2[a,b] = sum_up M[a,up] conj(M[b,up])

        so that summing psi_bra * conj(psi_ket) over the "spectator" spin
        index -- the expensive part of the original per-basis-state loop --
        becomes two matmuls, done once, instead of once per orbital pair.
        Each orbital-pair contribution is then a single vectorized sum over
        its (small) connectivity template, reusing the same fast_ops
        machinery as the hopping term.
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_single_particle_density_matrix)")
        M = np.asarray(psi_gs).reshape(self.dim_dn, self.dim_up)

        G = M.conj().T @ M    # (dim_up, dim_up)
        H2 = M @ M.conj().T   # (dim_dn, dim_dn)

        rho = np.zeros((NUM_ORBITALS, NUM_ORBITALS), dtype=np.complex128)

        diagG = np.diag(G).real   # diagG[up] = sum_dn |M[dn,up]|^2
        diagH2 = np.diag(H2).real  # diagH2[dn] = sum_up |M[dn,up]|^2

        for site in range(NUM_SITES):
            rho[site * SPINS + 0, site * SPINS + 0] = self._occ_up[site] @ diagG
            rho[site * SPINS + 1, site * SPINS + 1] = self._occ_dn[site] @ diagH2

        for i in range(NUM_SITES):
            for j in range(NUM_SITES):
                if i == j:
                    continue

                bra_idx, ket_idx, sign = self._get_pair_template_up(j, i)
                rho[j * SPINS + 0, i * SPINS + 0] = np.sum(sign * G[ket_idx, bra_idx])

                bra_idx, ket_idx, sign = self._get_pair_template_dn(j, i)
                rho[j * SPINS + 1, i * SPINS + 1] = np.sum(sign * H2[bra_idx, ket_idx])

        # up-down cross block stays exactly zero (spin-diagonal Hamiltonian)

        trace = np.trace(rho).real
        expected_n = N_UP + N_DN
        if abs(trace - expected_n) > 1e-6:
            warnings.warn(
                f"trace(rho) = {trace:.6f}, expected {expected_n}. "
                f"Check the electron sector / fermionic_sign convention.",
                RuntimeWarning,
            )

        return rho

    def compute_chern_number(
        self,
        delta: float,
        U: float,
        V: float,
        grid_size: int = 6,
        overlap_tol: float = 1e-6,
        verbose: bool = False,
    ) -> int:
        """
        Calculates spinful total Chern Number (C_total = C_up + C_down)
        using Fukui-Hatsugai-Suzuki gauge-invariant flux integration.

        Unchanged from the flat-basis version: this only ever needs the flat
        ground-state vectors and their overlaps <psi(theta)|psi(theta')>,
        which are agnostic to how the Hamiltonian/basis are internally
        represented.

        grid_size default raised from 4 -> 6: with only 4x4 plaquettes the
        discretized Berry curvature can alias badly near a topological
        transition. Re-run at a couple of grid sizes (e.g. 6, 8, 10) and
        confirm the integer is stable before trusting a single value.
        """
        N = grid_size
        flux_x_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
        flux_y_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)

        psi_grid = [[None for _ in range(N)] for _ in range(N)]

        v_warm = None
        for ix, fx in enumerate(flux_x_grid):
            for iy, fy in enumerate(flux_y_grid):
                try:
                    result = self.solver.solve(delta, U, V, flux_x=fx, flux_y=fy, v0=v_warm)
                except TypeError:
                    result = self.solver.solve(delta, U, V, flux_x=fx, flux_y=fy)

                _, psi = result
                psi_grid[ix][iy] = psi
                v_warm = psi

        Ux = np.zeros((N, N), dtype=np.complex128)
        Uy = np.zeros((N, N), dtype=np.complex128)

        for ix in range(N):
            ix_next = (ix + 1) % N
            for iy in range(N):
                iy_next = (iy + 1) % N

                overlap_x = np.vdot(psi_grid[ix][iy], psi_grid[ix_next][iy])
                overlap_y = np.vdot(psi_grid[ix][iy], psi_grid[ix][iy_next])

                if np.abs(overlap_x) < overlap_tol:
                    raise RuntimeError(
                        f"Near-zero overlap |<psi({ix},{iy})|psi({ix_next},{iy})>| = "
                        f"{np.abs(overlap_x):.2e} at flux point ({ix},{iy}) -> ({ix_next},{iy}). "
                        f"This usually means a (near-)degenerate ground state or a branch jump "
                        f"between eigensolver calls. Check the gap at this flux point and "
                        f"consider warm-starting / increasing grid_size."
                    )
                if np.abs(overlap_y) < overlap_tol:
                    raise RuntimeError(
                        f"Near-zero overlap |<psi({ix},{iy})|psi({ix},{iy_next})>| = "
                        f"{np.abs(overlap_y):.2e} at flux point ({ix},{iy}) -> ({ix},{iy_next}). "
                        f"This usually means a (near-)degenerate ground state or a branch jump "
                        f"between eigensolver calls. Check the gap at this flux point and "
                        f"consider warm-starting / increasing grid_size."
                    )

                Ux[ix, iy] = overlap_x / np.abs(overlap_x)
                Uy[ix, iy] = overlap_y / np.abs(overlap_y)

        total_curvature = 0.0
        max_abs_field_strength = 0.0
        for ix in range(N):
            ix_next = (ix + 1) % N
            for iy in range(N):
                iy_next = (iy + 1) % N

                plaquette = (
                    Ux[ix, iy] *
                    Uy[ix_next, iy] *
                    np.conj(Ux[ix, iy_next]) *
                    np.conj(Uy[ix, iy])
                )

                field_strength = np.angle(plaquette)
                total_curvature += field_strength
                max_abs_field_strength = max(max_abs_field_strength, abs(field_strength))

        if verbose and max_abs_field_strength > np.pi / 2:
            warnings.warn(
                f"Largest single-plaquette field strength is {max_abs_field_strength:.3f} rad "
                f"(> pi/2). Curvature may be varying too fast for grid_size={N} to resolve "
                f"correctly; try a finer grid.",
                RuntimeWarning,
            )

        raw = total_curvature / (2 * np.pi)
        chern = int(np.round(raw))

        if verbose and abs(raw - chern) > 0.05:
            warnings.warn(
                f"Chern sum {raw:.4f} is not close to an integer (rounded to {chern}). "
                f"This usually indicates gauge inconsistency (degeneracy/branch jump) or an "
                f"under-resolved flux grid.",
                RuntimeWarning,
            )

        return chern

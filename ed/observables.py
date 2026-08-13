"""
Observable module: Calculates CDW, squared SDW correlation,
and spinful Fukui-Hatsugai-Suzuki Chern Numbers.
"""

import warnings
import numpy as np
from .config import NUM_SITES, SPINS, NUM_ORBITALS, NUM_ELECTRONS
from .basis import BasisOperations, FockBasis
from .lattice import HoneycombLattice
from .ed_solver import ExactDiagonalizationSolver
from .fast_ops import build_orbital_pair_template


class ObservableCalculator:
    """
    Computes physical order parameters and topological invariants.
    """
    def __init__(self, basis: FockBasis, lattice: HoneycombLattice, solver: ExactDiagonalizationSolver):
        self.basis = basis
        self.lattice = lattice
        self.solver = solver
        self.fock_dim = len(basis.basis_states)

        # Precompute operator diagonal vectors for ultra-fast expectations
        self._site_density_ops = self._precompute_site_densities()

        # Lazy cache of orbital-pair connectivity templates (see fast_ops),
        # used to vectorize compute_single_particle_density_matrix. Keyed by
        # (orb_i, orb_j); built on first use and reused across ground states.
        self._pair_template_cache = {}

    def _get_pair_template(self, orb_i: int, orb_j: int):
        key = (orb_i, orb_j)
        template = self._pair_template_cache.get(key)
        if template is None:
            template = build_orbital_pair_template(self.basis.basis_states, orb_i, orb_j)
            self._pair_template_cache[key] = template
        return template

    def _precompute_site_densities(self) -> np.ndarray:
        """
        Precomputes n_{i,up} and n_{i,down} occupancy per site for all Fock states.
        """
        densities = np.zeros((NUM_SITES, SPINS, self.fock_dim), dtype=np.float64)

        for b_idx, state in enumerate(self.basis.basis_states):
            for site in range(NUM_SITES):
                if BasisOperations.is_occupied(state, site * SPINS + 0):
                    densities[site, 0, b_idx] = 1.0
                if BasisOperations.is_occupied(state, site * SPINS + 1):
                    densities[site, 1, b_idx] = 1.0

        return densities

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

    def compute_cdw(self, psi_gs: np.ndarray) -> float:
        """
        Computes staggered Charge-Density Wave structure factor

            S_CDW = (1/N) * sum_{i,j} xi_i * xi_j * <n_i n_j>,   n_i = n_i_up + n_i_down

        with xi_i either +1 or -1 depending on the sublattice index
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_cdw)")
        prob_density = np.abs(psi_gs) ** 2
        norm_sq = np.sum(prob_density)
        if norm_sq > 0:
            prob_density = prob_density / norm_sq  # guard against unnormalized input

        # Precompute total density operator (diagonal) per site: n_i = n_i_up + n_i_down
        n_ops = self._site_density_ops[:, 0, :] + self._site_density_ops[:, 1, :]  # (NUM_SITES, fock_dim)

        cdw_val = 0.0
        for i in range(NUM_SITES):
            xi_i = 1.0 if self.lattice.site_to_sublattice[i] == 0 else -1.0
            for j in range(NUM_SITES):
                xi_j = 1.0 if self.lattice.site_to_sublattice[j] == 0 else -1.0

                # Diagonal expectation value <psi| n_i n_j |psi>
                n_ij_diag = n_ops[i] * n_ops[j]
                exp_val = np.dot(n_ij_diag, prob_density)

                cdw_val += xi_i * xi_j * exp_val

        return float(cdw_val / NUM_SITES)

    def compute_sdw_squared(self, psi_gs: np.ndarray) -> float:
        """
        Computes the staggered Spin-Density Wave structure factor, matching
        Eq. (6) of Uria-Alvarez & Valenti (2026):

            S_SDW = (1/N) * sum_{i,j} xi_i * xi_j * <(n_i_up - n_i_dn)(n_j_up - n_j_dn)>

        Since (n_i_up - n_i_dn) = 2 * S_i^z, this equals
        (4/N) * sum_{i,j} xi_i * xi_j * <S_i^z S_j^z>.

        NOTE: despite the method name (kept for backwards compatibility with
        callers), the paper's S_SDW is NOT a square root of anything - it is
        the raw two-point structure factor (values up to O(N), see Fig. 2/6).
        The previous implementation here incorrectly took sqrt() of the sum,
        dropped the factor of 4 relating S^z S^z to (n_up-n_dn)(n_up-n_dn),
        and normalized by N *after* the sqrt instead of N multiplying the sum
        directly - all three combined made this not reproduce the paper's
        SDW observable.
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_sdw_squared)")
        prob_density = np.abs(psi_gs) ** 2
        norm_sq = np.sum(prob_density)
        if norm_sq > 0:
            prob_density = prob_density / norm_sq  # guard against unnormalized input

        sdw_val = 0.0

        for i in range(NUM_SITES):
            xi_i = 1.0 if self.lattice.site_to_sublattice[i] == 0 else -1.0
            nz_i = self._site_density_ops[i, 0] - self._site_density_ops[i, 1]  # n_i_up - n_i_dn

            for j in range(NUM_SITES):
                xi_j = 1.0 if self.lattice.site_to_sublattice[j] == 0 else -1.0
                nz_j = self._site_density_ops[j, 0] - self._site_density_ops[j, 1]

                # Diagonal expectation value <psi| (n_i_up - n_i_dn)(n_j_up - n_j_dn) |psi>
                nz_ij_diag = nz_i * nz_j
                exp_val = np.dot(nz_ij_diag, prob_density)

                sdw_val += xi_i * xi_j * exp_val

        return float(sdw_val / NUM_SITES)

    def compute_single_particle_density_matrix(self, psi_gs: np.ndarray) -> np.ndarray:
        """
        Compute single-particle density matrix rho_ij = <c_i^dagger c_j>.

        Vectorized: rho[j, i] = sum over bra states connected to a ket state
        by removing orbital i and adding orbital j, weighted by the
        fermionic sign and psi_bra * conj(psi_ket). This is exactly the
        same physics as the original per-state, per-orbital-pair Python
        loop, but the "which bra connects to which ket, with what sign"
        structure is precomputed once per (i, j) pair (fast_ops,
        NUM_ORBITALS^2 pairs total) and reused across ground states, and
        each pair's contribution is a single vectorized numpy reduction
        instead of an O(fock_dim) inner loop.
        """
        self._check_normalized(psi_gs, label="psi_gs (compute_single_particle_density_matrix)")

        rho = np.zeros((NUM_ORBITALS, NUM_ORBITALS), dtype=np.complex128)

        # Diagonal: rho_ii = <n_i>
        prob_amp = psi_gs
        for orb in range(NUM_ORBITALS):
            site, spin = orb // 2, orb % 2
            occ_mask = self._site_density_ops[site, spin].astype(bool)
            rho[orb, orb] = np.sum(occ_mask * np.abs(prob_amp) ** 2)

        # Off-diagonal: rho[j, i] = <c_j^dagger c_i>, i != j
        for i in range(NUM_ORBITALS):
            for j in range(NUM_ORBITALS):
                if i == j:
                    continue
                # matches original loop's naming: i = orbital removed from
                # bra, j = orbital added to make ket -> pass (orb_i=j, orb_j=i)
                # to match fermionic_sign(state_bra, i, j) convention exactly.
                bra_idx, ket_idx, sign = self._get_pair_template(j, i)
                rho[j, i] = np.sum(sign * psi_gs[bra_idx] * np.conj(psi_gs[ket_idx]))

        trace = np.trace(rho).real
        expected_n = NUM_ELECTRONS  # 12 at true half-filling (see config.py)
        if expected_n is not None and abs(trace - expected_n) > 1e-6:
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

        grid_size default raised from 4 -> 6: with only 4x4 plaquettes the
        discretized Berry curvature can alias badly near a topological
        transition. Re-run at a couple of grid sizes (e.g. 6, 8, 10) and
        confirm the integer is stable before trusting a single value.
        """
        N = grid_size
        flux_x_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
        flux_y_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)

        # Store wavefunctions on N x N boundary torus
        psi_grid = [[None for _ in range(N)] for _ in range(N)]

        # 1. Compute ground state wavefunctions over boundary flux torus.
        # Warm-start each solve from the previous flux point's eigenvector so the
        # solver tracks a single continuous branch across the torus instead of
        # potentially landing on an arbitrary vector within a (near-)degenerate
        # subspace at each point independently. This assumes
        # ExactDiagonalizationSolver.solve accepts a v0 kwarg for an iterative
        # solver (e.g. Lanczos/ARPACK); if solve() is a dense eigh-based solver,
        # v0 is unnecessary (dense diagonalization always returns the true
        # ground state) and the try/except below will just fall back cleanly.
        v_warm = None
        for ix, fx in enumerate(flux_x_grid):
            for iy, fy in enumerate(flux_y_grid):
                try:
                    result = self.solver.solve(delta, U, V, flux_x=fx, flux_y=fy, v0=v_warm)
                except TypeError:
                    # solver.solve doesn't support warm-starting; fall back
                    result = self.solver.solve(delta, U, V, flux_x=fx, flux_y=fy)

                _, psi = result
                psi_grid[ix][iy] = psi
                v_warm = psi

        # 2. Compute Link Variables U_x and U_y for the total many-body state
        Ux = np.zeros((N, N), dtype=np.complex128)
        Uy = np.zeros((N, N), dtype=np.complex128)

        for ix in range(N):
            ix_next = (ix + 1) % N
            for iy in range(N):
                iy_next = (iy + 1) % N

                # Inner product <psi(theta) | psi(theta + dtheta)>
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

        # 3. Sum lattice Field Strength F(theta) over grid plaquettes
        total_curvature = 0.0
        max_abs_field_strength = 0.0
        for ix in range(N):
            ix_next = (ix + 1) % N
            for iy in range(N):
                iy_next = (iy + 1) % N

                # Plaquette link product: U_x(k) * U_y(k + dx) * U_x(k + dy)^-1 * U_y(k)^-1
                plaquette = (
                    Ux[ix, iy] *
                    Uy[ix_next, iy] *
                    np.conj(Ux[ix, iy_next]) *
                    np.conj(Uy[ix, iy])
                )

                # Principal branch logarithm Im(ln(P)) in (-pi, pi]
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

        # Total charge Chern number C_total = (1 / 2pi) * sum(F)
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
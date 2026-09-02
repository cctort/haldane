"""
Order parameters (CDW, SDW), the single-particle density matrix, and the
Chern number.

CDW/SDW are staggered density-density sums, the same kind of object as
the on-site U term in the review (Eq. 14 is <n_up n_dn> on one site;
here we sum <n_i n_j> with a staggered sign over all site pairs), just
built from |psi|^2 on the (dn, up) probability grid instead of one
matrix element.
"""
import numpy as np
from .lattice import NUM_SITES, SUB_SIGN
from .basis import BASIS
from .fast_ops import occupied, hop_matrix_elements


def occupation_table(states):
    return np.array([[occupied(s, site) for s in states] for site in range(NUM_SITES)], dtype=float)


class Observables:
    def __init__(self, solver):
        self.solver = solver
        self.occ_up = occupation_table(BASIS.up.states)
        self.occ_dn = self.occ_up if BASIS.dn is BASIS.up else occupation_table(BASIS.dn.states)
        self.stag_up = SUB_SIGN @ self.occ_up
        self.stag_dn = SUB_SIGN @ self.occ_dn

    def _prob(self, psi):
        """|psi|^2 reshaped to (dim_dn, dim_up), normalized."""
        prob = np.abs(psi.reshape(BASIS.dim_dn, BASIS.dim_up)) ** 2
        return prob / prob.sum()

    def cdw(self, psi):
        """Staggered charge structure factor S_CDW = (1/N) sum_ij xi_i xi_j <n_i n_j>."""
        prob = self._prob(psi)
        stag_charge = self.stag_dn[:, None] + self.stag_up[None, :]
        return float(np.sum(stag_charge * stag_charge * prob) / NUM_SITES)

    def sdw(self, psi):
        """Staggered spin structure factor, same idea with n_up - n_dn."""
        prob = self._prob(psi)
        stag_spin = self.stag_up[None, :] - self.stag_dn[:, None]
        return float(np.sum(stag_spin * stag_spin * prob) / NUM_SITES)

    def density_matrix(self, psi):
        """
        rho_ij = <c_i^dagger c_j> per spin (block-diagonal in spin, since
        H never mixes spins). Uses the Gram-matrix trick to sum over the
        spectator spin index in one matmul instead of a loop.
        """
        M = psi.reshape(BASIS.dim_dn, BASIS.dim_up)
        G = M.conj().T @ M     # (dim_up, dim_up)
        H2 = M @ M.conj().T    # (dim_dn, dim_dn)

        rho_up = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)
        rho_dn = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)

        diagG, diagH2 = np.diag(G).real, np.diag(H2).real
        for site in range(NUM_SITES):
            rho_up[site, site] = self.occ_up[site] @ diagG
            rho_dn[site, site] = self.occ_dn[site] @ diagH2

        for i in range(NUM_SITES):
            for j in range(NUM_SITES):
                if i == j:
                    continue
                for bra, ket, sign in hop_matrix_elements(BASIS.up.states, j, i):
                    rho_up[j, i] += sign * G[ket, bra]
                for bra, ket, sign in hop_matrix_elements(BASIS.dn.states, j, i):
                    rho_dn[j, i] += sign * H2[bra, ket]

        return rho_up, rho_dn

    def chern_number(self, delta, U, V, grid=4):
        """
        Total (up+down) Chern number via Fukui-Hatsugai-Suzuki flux
        integration: build the ground state on an NxN flux grid, form
        gauge-invariant link variables between neighboring points, and
        sum the discretized Berry curvature over all plaquettes.
        """
        angles = np.linspace(0, 2 * np.pi, grid, endpoint=False)
        psi = [[None] * grid for _ in range(grid)]

        v0 = None
        for ix, fx in enumerate(angles):
            for iy, fy in enumerate(angles):
                _, p = self.solver.ground_state(delta, U, V, fx, fy, v0)
                psi[ix][iy] = p
                v0 = p.reshape(-1, 1)

        def link(a, b):
            ov = np.vdot(a, b)
            return ov / abs(ov)

        curvature = 0.0
        for ix in range(grid):
            ix2 = (ix + 1) % grid
            for iy in range(grid):
                iy2 = (iy + 1) % grid
                plaquette = (
                    link(psi[ix][iy], psi[ix2][iy])
                    * link(psi[ix2][iy], psi[ix2][iy2])
                    * np.conj(link(psi[ix][iy2], psi[ix2][iy2]))
                    * np.conj(link(psi[ix][iy], psi[ix][iy2]))
                )
                curvature += np.angle(plaquette)

        return curvature / (2 * np.pi)

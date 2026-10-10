import numpy as np
from scipy import sparse
from .basis import SpinfulBasis
from .lattice import Lattice
from .hamiltonian import hop_element, occupation_table
from .config import Config
from .ed_solver import EDSolver


class Observables:
    def __init__(self, cfg: Config, basis: SpinfulBasis, lattice: Lattice, solver: EDSolver):
        self.solver = solver
        self.basis = basis
        self.lattice = lattice

        self.occ_up = occupation_table(basis.up.states, lattice.n_sites)
        self.occ_dn = self.occ_up if basis.dn is basis.up else occupation_table(basis.dn.states, lattice.n_sites)
        self.stag_up = lattice.signs @ self.occ_up
        self.stag_dn = lattice.signs @ self.occ_dn

        self.n_flux = cfg.n_flux

    def prob(self, psi):
        """|psi|^2 reshaped to (dim_dn, dim_up), normalized."""
        prob = np.abs(psi.reshape(self.basis.dim_dn, self.basis.dim_up)) ** 2
        return prob / prob.sum()

    def corr_ch(self, psi):
        """ Spatial charge correlation elements <n_i n_j> / N."""
        prob = self.prob(psi)  # shape: (dim_dn, dim_up)

        p_up = prob.sum(axis=0)  # shape: (dim_up,)
        p_dn = prob.sum(axis=1)  # shape: (dim_dn,)

        n_up_n_up = (self.occ_up * p_up) @ self.occ_up.T
        n_dn_n_dn = (self.occ_dn * p_dn) @ self.occ_dn.T
        n_up_n_dn = self.occ_up @ prob.T @ self.occ_dn.T

        return (n_up_n_up + n_up_n_dn + n_up_n_dn.T + n_dn_n_dn) / self.lattice.n_sites

    def corr_sz(self, psi):
        """ Spatial spin correlation elements <Sz_i Sz_j> / N. """
        prob = self.prob(psi)  # shape: (dim_dn, dim_up)

        p_up = prob.sum(axis=0)  # shape: (dim_up,)
        p_dn = prob.sum(axis=1)  # shape: (dim_dn,)

        n_up_n_up = (self.occ_up * p_up) @ self.occ_up.T
        n_dn_n_dn = (self.occ_dn * p_dn) @ self.occ_dn.T
        n_up_n_dn = self.occ_up @ prob.T @ self.occ_dn.T

        return (n_up_n_up - n_up_n_dn - n_up_n_dn.T + n_dn_n_dn) / (4 * self.lattice.n_sites)
    
    def hopping_op(self, states, i, j):
        """Build sparse operator matrix for c^\dagger_i c_j using hop_element."""
        dim = len(states)
        rows, cols, vals = [], [], []
        for bra, ket, sign in hop_element(states, i, j):
            rows.append(bra)
            cols.append(ket)
            vals.append(sign)
        return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()

    def corr_sx(self, psi):
        """Spatial transverse spin correlation elements <Sx_i Sx_j> / N."""
        dim_dn, dim_up = self.basis.dim_dn, self.basis.dim_up
        M = psi.reshape(dim_dn, dim_up)
        n_sites = self.lattice.n_sites
        corr_sx_mat = np.zeros((n_sites, n_sites), dtype=float)

        for i in range(n_sites):
            for j in range(n_sites):
                # <S+_i S-_j> = <c^\dagger_{i,up} c_{i,dn} c^\dagger_{j,dn} c_{j,up}>
                up_plus = self.hopping_op(self.basis.up.states, i, j)
                dn_plus_dag = self.hopping_op(self.basis.dn.states, j, i)
                new_M_plus = dn_plus_dag.conj().T @ M @ up_plus
                term1 = np.vdot(M, new_M_plus).real

                # <S-_i S+_j> = <c^\dagger_{i,dn} c_{i,up} c^\dagger_{j,up} c_{j,dn}>
                dn_minus = self.hopping_op(self.basis.dn.states, i, j)
                up_minus = self.hopping_op(self.basis.up.states, j, i)
                new_M_minus = dn_minus @ M @ up_minus
                term2 = np.vdot(M, new_M_minus).real

                corr_sx_mat[i, j] = (term1 + term2)

        return corr_sx_mat / (4 * n_sites)

    def S_q0(self, corr):
        return np.sum(corr)

    def stagger(self, corr_mat):
        stagg_mat = self.lattice.signs[:, None] * self.lattice.signs[None, :]
        return stagg_mat * corr_mat

    def S_qpi(self, corr):
        stagg_corr = self.stagger(corr)
        return np.sum(stagg_corr)

    def density_matrix(self, psi):
        """
        rho_ij = <c_i^dagger c_j> per spin.
        """
        M = psi.reshape(self.basis.dim_dn, self.basis.dim_up)
        G = M.conj().T @ M     # (dim_up, dim_up)
        H2 = M @ M.conj().T    # (dim_dn, dim_dn)

        rho_up = np.zeros((self.lattice.n_sites, self.lattice.n_sites), dtype=complex)
        rho_dn = np.zeros((self.lattice.n_sites, self.lattice.n_sites), dtype=complex)

        diagG, diagH2 = np.diag(G).real, np.diag(H2).real
        for site in range(self.lattice.n_sites):
            rho_up[site, site] = self.occ_up[site] @ diagG
            rho_dn[site, site] = self.occ_dn[site] @ diagH2

        for i in range(self.lattice.n_sites):
            for j in range(self.lattice.n_sites):
                if i == j:
                    continue
                for bra, ket, sign in hop_element(self.basis.up.states, j, i):
                    rho_up[j, i] += sign * G[ket, bra]
                for bra, ket, sign in hop_element(self.basis.dn.states, j, i):
                    rho_dn[j, i] += sign * H2[bra, ket]

        return rho_up, rho_dn

    def chern_number(self, delta, U, V, site_dis=None, bond_dis=None):
        """
        Total (up+down) Chern number via Fukui-Hatsugai-Suzuki flux
        integration.
        """
        angles = np.linspace(0, 2 * np.pi, self.n_flux, endpoint=False)
        psi = [[None] * self.n_flux for _ in range(self.n_flux)]

        v0 = None
        for ix, fx in enumerate(angles):
            for iy, fy in enumerate(angles):
                _, p = self.solver.ground_state(delta, U, V, fx, fy, v0, False, site_dis, bond_dis)
                psi[ix][iy] = p
                v0 = p

        def phase(a, b):
            braket = np.vdot(a, b)
            return braket / abs(braket)

        curvature = 0.0
        for ix in range(self.n_flux):
            ix2 = (ix + 1) % self.n_flux
            for iy in range(self.n_flux):
                iy2 = (iy + 1) % self.n_flux
                plaquette = (
                    phase(psi[ix][iy], psi[ix2][iy])
                    * phase(psi[ix2][iy], psi[ix2][iy2])
                    * np.conj(phase(psi[ix][iy2], psi[ix2][iy2]))
                    * np.conj(phase(psi[ix][iy], psi[ix][iy2]))
                )
                curvature += np.angle(plaquette)

        return curvature / (2 * np.pi)
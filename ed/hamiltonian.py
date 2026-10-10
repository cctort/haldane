import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .config import Config
from .lattice import Lattice
from .basis import SpinfulBasis


def occupied(state, orb):
    return (state >> orb) & 1 # shifts the orb-th bit to the first position and checks if 1


def hop_element(states, i, j):
    """
    Stores all finite <bra|c_i^dagger c_j|ket> matrix elements, 
    which are either 1 or -1, depending if the sites in between 
    i and j are even or odd.
    """
    index = {s: k for k, s in enumerate(states)}
    lo, hi = (i, j) if i < j else (j, i)
    between_mask = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1) # bits between lo and hi are 1, rest are zero

    entries = []
    for ket, init_state in enumerate(states):
        if not occupied(init_state, j):
            continue
        tmp_state = init_state & ~(1 << j) # init_state AND NOT (1 << j)
        if occupied(tmp_state, i):
            continue
        final_state = tmp_state | (1 << i) # tmp_state OR (1 << i)
        bra = index[final_state]
        sign = -1 if bin(init_state & between_mask).count("1") % 2 else 1 # -1 if odd number of 1s between i and j, else +1
        entries.append((ket, bra, sign))
    return entries


def hopping_matrix(states, lattice: Lattice, t1, t2, phi, flux_x=0.0, flux_y=0.0, bond_dis=None):
    """Spin-resolved hopping matrix c_i^dagger c_j + h.c. for each flux value."""
    dim = len(states)
    rows, cols, vals = [], [], []

    for i, j, shift in lattice.bonds[1]:
        n1, n2 = shift
        dt = bond_dis.get((i, j), 0.0) if bond_dis else 0.0
        amp = - (t1 + dt) * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_element(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    for i, j, shift in lattice.bonds[2]:
        n1, n2 = shift
        nu = lattice.chirality(i, j, shift)
        dt2 = bond_dis.get((i, j), 0.0) if bond_dis else 0.0
        amp = - (t2 + dt2) * np.exp(1j * np.pi * nu * phi) * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_element(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()


def occupation_table(states, n_sites):
    """<j|n_i|j> for all sites i and states j, stored as a (N_S, dim) array."""
    return np.array([[occupied(s, site) for s in states] for site in range(n_sites)], dtype=float)


def diagonal(lattice, occ_dn, occ_up, delta, U, V, site_dis=None):
    """(dim_dn, dim_up) diagonal of H from the staggered potential, U, V."""
    diag = np.zeros((occ_dn.shape[1], occ_up.shape[1]))

    if delta:
        stag_dn = lattice.signs @ occ_dn
        stag_up = lattice.signs @ occ_up
        diag += delta * (stag_dn[:, None] + stag_up[None, :])

    if U:
        diag += U * (occ_dn.T @ occ_up)

    if V:
        for i, j, _ in lattice.bonds[1]:
            n_i = occ_dn[i][:, None] + occ_up[i][None, :]
            n_j = occ_dn[j][:, None] + occ_up[j][None, :]
            diag += V * n_i * n_j

    if site_dis is not None:
        dis_dn = site_dis @ occ_dn
        dis_up = site_dis @ occ_up
        diag += (dis_dn[:, None] + dis_up[None, :])

    return diag


class Hamiltonian:
    def __init__(self, cfg: Config, basis: SpinfulBasis, lattice: Lattice):
        self.basis = basis
        self.lattice = lattice
        self.t1 = cfg.t1
        self.t2 = cfg.t2
        self.phi = cfg.phi
        self.occ_up = occupation_table(basis.up.states, lattice.n_sites)
        self.occ_dn = self.occ_up if basis.dn is basis.up else occupation_table(basis.dn.states, lattice.n_sites)

    def build(self, delta, U, V, flux_x=0.0, flux_y=0.0, site_dis=None, bond_dis=None):
        H_up = hopping_matrix(self.basis.up.states, self.lattice, self.t1, self.t2, self.phi, flux_x, flux_y, bond_dis)
        H_dn = H_up if self.basis.dn is self.basis.up else hopping_matrix(self.basis.dn.states, self.lattice, self.t1, self.t2, self.phi, flux_x, flux_y, bond_dis)
        H_up_T = H_up.T.tocsr()
    
        diag = diagonal(self.lattice, self.occ_dn, self.occ_up, delta, U, V, site_dis)
        dim_up, dim_dn, dim = self.basis.dim_up, self.basis.dim_dn, self.basis.dim

        def matvec(v):
            M = v.reshape(dim_dn, dim_up) # len(v) = dim = dim_up * dim_dn
            out = H_dn @ M + M @ H_up_T + diag * M # Same as (H_up \otimes id_dn + id_up \otimes H_dn + diag) v
            return out.reshape(dim)

        return LinearOperator(shape=(dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.complex128)
import numpy as np
from functools import lru_cache
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .config import T1, T2, PHI, DTYPE
from .lattice import LATTICE, NUM_SITES, SUB_SIGN
from .basis import BASIS


def occupied(state, orb):
    return (state >> orb) & 1 # checks if orb is 1


def hop_matrix(states, i, j):
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
        tmp = init_state & ~(1 << j) # sets bit j to 0
        if occupied(tmp, i):
            continue
        final_state = tmp | (1 << i) # sets bit i to 1
        bra = index[final_state]
        sign = -1 if bin(init_state & between_mask).count("1") % 2 else 1
        entries.append((ket, bra, sign))
    return entries


def hopping_matrix(states, flux_x, flux_y):
    """Spin-resolved hopping matrix c_i^dagger c_j + h.c. for each flux value."""
    dim = len(states)
    rows, cols, vals = [], [], []

    for i, j, shift in LATTICE.nn_bonds:
        n1, n2 = shift
        amp = - T1 * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_matrix(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    for i, j, shift in LATTICE.nnn_bonds:
        n1, n2 = shift
        nu = LATTICE.chirality(i, j, shift)
        amp = - T2 * np.exp(1j * nu * PHI) * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_matrix(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=DTYPE).tocsr()


def occupation_table(states):
    """<j|n_i|j> for all sites i and states j, stored as a (NUM_SITES, dim) array."""
    return np.array([[occupied(s, site) for s in states] for site in range(NUM_SITES)], dtype=float)


def diagonal(occ_dn, occ_up, delta, U, V):
    """(dim_dn, dim_up) diagonal of H from the staggered potential, U, V."""
    diag = np.zeros((occ_dn.shape[1], occ_up.shape[1]))

    if delta:
        stag_dn = SUB_SIGN @ occ_dn
        stag_up = SUB_SIGN @ occ_up
        diag += delta * (stag_dn[:, None] + stag_up[None, :])

    if U:
        diag += U * (occ_dn.T @ occ_up)

    if V:
        for i, j, _ in LATTICE.nn_bonds:
            n_i = occ_dn[i][:, None] + occ_up[i][None, :]
            n_j = occ_dn[j][:, None] + occ_up[j][None, :]
            diag += V * n_i * n_j

    return diag


class Hamiltonian:
    def __init__(self):
        self.basis = BASIS
        self.occ_up = occupation_table(BASIS.up.states)
        self.occ_dn = self.occ_up if BASIS.dn is BASIS.up else occupation_table(BASIS.dn.states)

        self._get_diagonal = lru_cache(maxsize=1)(
            lambda delta, U, V: diagonal(self.occ_dn, self.occ_up, delta, U, V)
        )

    def build(self, delta, U, V, flux_x=0.0, flux_y=0.0):
        H_up = hopping_matrix(self.basis.up.states, flux_x, flux_y)
        H_dn = H_up if self.basis.dn is self.basis.up else hopping_matrix(self.basis.dn.states, flux_x, flux_y)
        H_up_T = H_up.T.tocsr()
        diag = self._get_diagonal(delta, U, V)

        dim_up, dim_dn, dim = self.basis.dim_up, self.basis.dim_dn, self.basis.dim

        def matvec(v):
            M = v.reshape(dim_dn, dim_up) # len(v) = dim = dim_up * dim_dn
            out = H_dn @ M + M @ H_up_T + diag * M # Same as (H_up \otimes id_dn + id_up \otimes H_dn + diag) v
            return out.reshape(dim)

        return LinearOperator(shape=(dim, dim), matvec=matvec, rmatvec=matvec, dtype=DTYPE)

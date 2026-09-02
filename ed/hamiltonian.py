"""
Haldane-Hubbard Hamiltonian, spin-factorized.

Since t1, t2, the Haldane phase and the twisted-boundary flux never mix
spins, and every interaction term is density-density,

    H = H_up (x) I_dn + I_up (x) H_dn + diag(delta, U, V)

H_up and H_dn are small (dim_up x dim_up, at most 924x924 here), built
with plain loops over lattice bonds -- exactly the same procedure the
review uses to build Ht and HU for its 6-state example (Eqs. 20-23),
just with a Haldane phase and a twisted-boundary flux folded into each
hopping amplitude. H is applied matrix-free (as a LinearOperator) by
reshaping a wavefunction into a (dim_dn, dim_up) array, so the full
dim_dn*dim_up matrix is never assembled.
"""
import numpy as np
from functools import lru_cache
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .config import T1, T2, PHI, DTYPE
from .lattice import LATTICE, NUM_SITES, SUB_SIGN
from .basis import BASIS
from .fast_ops import hop_matrix_elements, occupied


def spin_hopping_matrix(states, flux_x, flux_y):
    """Small single-spin hopping matrix c_i^dagger c_j + h.c. for one flux."""
    dim = len(states)
    rows, cols, vals = [], [], []

    for i, j, shift in LATTICE.nn_bonds:
        n1, n2 = shift
        amp = T1 * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_matrix_elements(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    for i, j, shift in LATTICE.nnn_bonds:
        n1, n2 = shift
        nu = LATTICE.chirality(i, j, shift)
        amp = T2 * np.exp(1j * nu * PHI) * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        for bra, ket, sign in hop_matrix_elements(states, i, j):
            rows += [ket, bra]
            cols += [bra, ket]
            vals += [sign * amp, sign * np.conj(amp)]

    return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=DTYPE).tocsr()


def occupation_table(states):
    """(NUM_SITES, dim) 0/1 occupation array, one row per site."""
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
        # diagonal(delta, U, V) doesn't depend on flux, but build() is called
        # once per flux point in flux_sweep/chern_number -- cache it (last
        # value only) so a sweep over many flux points at fixed (delta, U, V)
        # only pays for this once instead of once per point.
        self._get_diagonal = lru_cache(maxsize=1)(
            lambda delta, U, V: diagonal(self.occ_dn, self.occ_up, delta, U, V)
        )

    def build(self, delta, U, V, flux_x=0.0, flux_y=0.0):
        H_up = spin_hopping_matrix(self.basis.up.states, flux_x, flux_y)
        H_dn = H_up if self.basis.dn is self.basis.up else spin_hopping_matrix(self.basis.dn.states, flux_x, flux_y)
        H_up_T = H_up.T.tocsr()
        diag = self._get_diagonal(delta, U, V)

        dim_up, dim_dn, dim = self.basis.dim_up, self.basis.dim_dn, self.basis.dim

        def matvec(v):
            M = v.reshape(dim_dn, dim_up)
            out = H_dn @ M + M @ H_up_T + diag * M
            return out.reshape(dim)

        # H is Hermitian by construction, so the adjoint action is matvec itself.
        return LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=DTYPE)

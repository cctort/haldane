"""
Hamiltonian construction for the Haldane-Hubbard model, factorized by Sz
sector (see basis.py) and applied matrix-free via a scipy LinearOperator.

Why matrix-free
----------------
The combined Hilbert space has dimension dim_dn * dim_up (854,076 at
N_up=N_dn=6 on the 12A cluster). Explicitly assembling a sparse operator of
that size costs real memory and time on every call. But since the model
never mixes spin, the Hamiltonian has the structure

    H = H_up (x) I_dn + I_up (x) H_dn + D

where H_up/H_dn are small (dim_up x dim_up / dim_dn x dim_dn) single-spin
hopping matrices and D is a diagonal in the (dn, up) grid. Writing the
wavefunction as a (dim_dn, dim_up) array M (instead of a flat fock_dim
vector), the action of each term is:

    (I_up (x) H_dn) M  =  H_dn @ M
    (H_up (x) I_dn) M  =  M @ H_up.T
    D M                =  D * M   (elementwise)

so a matvec is just two small sparse matmuls plus an elementwise multiply --
no fock_dim x fock_dim matrix is ever built. H_up and H_dn only depend on
the lattice/flux/phase, not on the many-body sector size, so they stay small
(dim_up x dim_up, dim_dn x dim_dn) regardless of how big the combined space
gets.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .config import (
    NUM_SITES, T1, T2, PHI, DTYPE, DTYPE_REAL
)
from .lattice import get_lattice
from .basis import get_fock_basis
from .fast_ops import build_orbital_pair_template, build_occupation_masks


class HaldaneHubbardHamiltonian:
    """
    Build and manage the (Sz-factorized, matrix-free) Haldane-Hubbard
    Hamiltonian.

    One-time (parameter-independent) precomputation done in __init__:
      - hopping connectivity/sign templates for the up and down single-spin
        sub-bases (fast_ops, same trick as before but over a basis of size
        NUM_SITES=12 instead of NUM_ORBITALS=24 -- no more spin loop either,
        since hopping is spin-diagonal with identical amplitudes for both
        spins)
      - per-site occupation masks for each sub-basis, used to build the
        (dim_dn, dim_up) diagonal for delta/U/V without any Python loop over
        basis states.

    Every build_full_hamiltonian(...) call after that only touches
    NUM_SITES-scale bond loops and (dim_dn, dim_up)-scale array ops -- it
    never allocates anything of size fock_dim x fock_dim.
    """

    def __init__(self):
        self.lattice = get_lattice()
        self.basis = get_fock_basis()
        self.dim_up = self.basis.dim_up
        self.dim_dn = self.basis.dim_dn
        self.fock_dim = self.basis.fock_dim

        same_sector = self.basis.dn is self.basis.up

        # Hopping connectivity, one-time, per spin sub-basis.
        self._hop_templates_up = self._build_hop_templates(self.basis.up.states_np)
        self._hop_templates_dn = (self._hop_templates_up if same_sector
                                   else self._build_hop_templates(self.basis.dn.states_np))

        # Per-site occupation masks (NUM_SITES, dim), one-time, per sub-basis.
        self._occ_up = build_occupation_masks(self.basis.up.states_np, NUM_SITES).astype(DTYPE_REAL)
        self._occ_dn = (self._occ_up if same_sector
                         else build_occupation_masks(self.basis.dn.states_np, NUM_SITES).astype(DTYPE_REAL))

        # Sublattice sign vector, used for the staggered potential.
        self._sub_sign = np.array(
            [1.0 if self.lattice.site_to_sublattice[s] == 0 else -1.0 for s in range(NUM_SITES)],
            dtype=DTYPE_REAL,
        )

    # ------------------------------------------------------------------
    # One-time structural precomputation
    # ------------------------------------------------------------------
    def _build_hop_templates(self, states_np: np.ndarray) -> Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        templates = {}
        for bond_type, bonds in (('nn', self.lattice.get_bonds('nn')),
                                  ('nnn', self.lattice.get_bonds('nnn'))):
            for site_i, site_j in bonds:
                templates[(bond_type, site_i, site_j)] = build_orbital_pair_template(states_np, site_i, site_j)
        return templates

    # ------------------------------------------------------------------
    # Per-call (parameter-dependent) assembly -- all cheap from here on
    # ------------------------------------------------------------------
    def _build_spin_hopping(self, templates, dim: int, flux_x: float, flux_y: float) -> sparse.csr_matrix:
        """Small (dim x dim) single-spin hopping matrix for the given flux."""
        tbc_nn = self._get_tbc_phases(flux_x, flux_y, 'nn')
        tbc_nnn = self._get_tbc_phases(flux_x, flux_y, 'nnn')
        hald_phases = self._get_chiralities()

        rows, cols, data = [], [], []
        for (bond_type, site_i, site_j), (bra_idx, ket_idx, sign) in templates.items():
            if bond_type == 'nn':
                t_eff = T1 * tbc_nn[(site_i, site_j)]
            else:
                t_eff = T2 * hald_phases[(site_i, site_j)] * tbc_nnn[(site_i, site_j)]

            vals = sign * t_eff
            rows.append(ket_idx); cols.append(bra_idx); data.append(vals)
            rows.append(bra_idx); cols.append(ket_idx); data.append(sign * np.conj(t_eff))

        H = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(dim, dim), dtype=DTYPE
        )
        return H.tocsr()

    def _staggered_vector(self, occ: np.ndarray) -> np.ndarray:
        """sum_i sign_i * n_i, per single-spin basis state. occ: (NUM_SITES, dim)."""
        return self._sub_sign @ occ  # (dim,)

    def build_diagonal(self, delta: float, U: float, V: float) -> np.ndarray:
        """
        (dim_dn, dim_up) real array: the diagonal of H in the combined basis,
        from the staggered potential, Hubbard U, and NN Coulomb V terms.
        Built entirely from the one-time occupation masks -- no loop over
        the combined (854k-entry) basis.
        """
        diag = np.zeros((self.dim_dn, self.dim_up), dtype=DTYPE_REAL)

        if delta != 0:
            stag_up = self._staggered_vector(self._occ_up)
            stag_dn = stag_up if self._occ_dn is self._occ_up else self._staggered_vector(self._occ_dn)
            diag += delta * (stag_dn[:, None] + stag_up[None, :])

        if U != 0:
            # diag_hub[dn, up] = sum_i occ_dn[i, dn] * occ_up[i, up]
            diag += U * (self._occ_dn.T @ self._occ_up)

        if V != 0:
            for site_i, site_j in self.lattice.get_bonds('nn'):
                n_i = self._occ_dn[site_i][:, None] + self._occ_up[site_i][None, :]
                n_j = self._occ_dn[site_j][:, None] + self._occ_up[site_j][None, :]
                diag += V * (n_i * n_j)

        return diag

    def build_full_hamiltonian(
        self, delta: float, U: float, V: float, flux_x: float = 0.0, flux_y: float = 0.0
    ) -> LinearOperator:
        """
        Return H as a scipy LinearOperator acting on flat fock_dim vectors
        (internally reshaped to (dim_dn, dim_up)) -- eigsh accepts this
        directly, exactly like it accepted the old sparse matrix, but no
        fock_dim x fock_dim matrix is ever materialized.
        """
        H_up = self._build_spin_hopping(self._hop_templates_up, self.dim_up, flux_x, flux_y)
        same_sector = self._hop_templates_dn is self._hop_templates_up
        H_dn = H_up if same_sector else self._build_spin_hopping(self._hop_templates_dn, self.dim_dn, flux_x, flux_y)

        H_up_T = H_up.T.tocsr()
        diag = self.build_diagonal(delta, U, V)

        dim_up, dim_dn, fock_dim = self.dim_up, self.dim_dn, self.fock_dim

        def matvec(v):
            M = np.asarray(v).reshape(dim_dn, dim_up)
            out = H_dn @ M            # I_up (x) H_dn
            out = out + M @ H_up_T    # H_up (x) I_dn
            out = out + diag * M      # diagonal terms
            return out.reshape(fock_dim)

        def rmatvec(v):
            # H is Hermitian by construction (hopping built as c^dag c + h.c.,
            # diagonal real), so the adjoint action is the same map.
            return matvec(v)

        return LinearOperator((fock_dim, fock_dim), matvec=matvec, rmatvec=rmatvec, dtype=DTYPE)

    # ------------------------------------------------------------------
    # Twisted boundary phases / Haldane chirality (unchanged physics,
    # just no longer looped over a spin index since it doesn't depend on
    # spin at all)
    # ------------------------------------------------------------------
    def _get_tbc_phases(self, flux_x: float, flux_y: float, bond_type: str) -> Dict[Tuple[int, int], complex]:
        tbc_phases = {}
        bond_shifts = self.lattice.get_bond_shift(bond_type)
        for bond, shift in bond_shifts.items():
            n1, n2 = shift
            tbc_phases[bond] = np.exp(1j * (n1 * flux_x + n2 * flux_y))
        return tbc_phases

    def _get_chiralities(self) -> Dict[Tuple[int, int], complex]:
        hald_phases = {}
        bond_shifts = self.lattice.get_bond_shift('nnn')
        for bond, shift in bond_shifts.items():
            pos_i = self.lattice.sites[bond[0]].pos
            pos_j = self.lattice.sites[bond[1]].pos
            n1, n2 = shift

            shift_vec = n1 * self.lattice.L1 + n2 * self.lattice.L2
            dx, dy = (pos_j[0] + shift_vec[0]) - pos_i[0], (pos_j[1] + shift_vec[1]) - pos_i[1]
            sub = self.lattice.sites[bond[0]].sublattice
            angle = np.arctan2(dy, dx)

            if sub == 0:
                nu = 1.0 if np.sin(3 * angle) > 0 else -1.0
            else:
                nu = -1.0 if np.sin(3 * angle) > 0 else 1.0

            hald_phases[bond] = np.exp(1j * nu * PHI)
        return hald_phases


class HamiltonianFactory:
    def __init__(self):
        self._instance = None

    def get_hamiltonian(self, rebuild: bool = False) -> HaldaneHubbardHamiltonian:
        if self._instance is None or rebuild:
            self._instance = HaldaneHubbardHamiltonian()
        return self._instance


_FACTORY = HamiltonianFactory()


def get_hamiltonian() -> HaldaneHubbardHamiltonian:
    return _FACTORY.get_hamiltonian()

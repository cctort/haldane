"""
Hamiltonian construction for the Haldane-Hubbard model.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import sparse
from typing import Optional
from .config import (
    NUM_SITES, SPINS, NUM_ORBITALS, 
    T1, T2, PHI, DTYPE, DTYPE_REAL
)
from .lattice import get_lattice
from .basis import get_fock_basis, BasisOperations
from .fast_ops import build_orbital_pair_template, build_occupation_masks


class HaldaneHubbardHamiltonian:
    """
    Build and manage the Haldane-Hubbard Hamiltonian.

    Performance note: the connectivity (which basis index pairs a hop
    connects) and fermionic sign of every hopping bond, and the occupation
    pattern of every orbital, depend only on the lattice/basis structure --
    not on delta, U, V, or the twisted-boundary flux. Those are therefore
    precomputed once in __init__ (_build_hopping_templates /
    _build_occupation_masks) via vectorized numpy operations
    (see fast_ops.py), instead of being recomputed with a Python loop over
    the full Fock basis on every call to build_full_hamiltonian. Each call
    then only has to combine cached arrays with the current scalar
    parameters, which is what makes repeated calls (e.g. one per flux point
    in compute_chern_number, or one per point in a phase-diagram sweep)
    cheap.
    """
    def __init__(self):
        self.lattice = get_lattice()
        self.basis = get_fock_basis()
        self.fock_dim = len(self.basis.basis_states)

        # One-time, parameter-independent precomputation.
        self._occ = build_occupation_masks(self.basis.basis_states, NUM_ORBITALS)
        self._build_hopping_templates()
        
    def build_full_hamiltonian(
            self, delta: float, U: float, V: float,flux_x: float = 0.0, flux_y: float = 0.0
        ) -> sparse.csr_matrix:
        """
        Build full Hermitian Hamiltonian H = H_0 + H_int.
        """
        H0_hop = self._build_hopping(flux_x, flux_y)
        H0_stag = self._build_staggered_potential(delta)
        H_hubbard = self._build_hubbard(U)
        H_coulomb = self._build_coulomb(V)
        
        H = H0_hop + H0_stag + H_hubbard + H_coulomb
        return H.tocsr()
    
    def _get_tbc_phases(
            self, flux_x: float, flux_y: float, bond_type: str
        ) -> Dict[Tuple[int, int], float]:
        """
        Get the twisted-boundary-condition phase for every bond
        in the same format as the corresponding lattice shifts.
        """
        tbc_phases = {}
        bond_shifts = self.lattice.get_bond_shift(bond_type)
        for bond, shift in bond_shifts.items():
            n1, n2 = shift
            tbc_phases[bond] = np.exp(1j * (n1 * flux_x + n2 * flux_y))

        return tbc_phases

    def _get_chiralities(self) -> Dict[Tuple[int, int], int]:
        """
        Determine Haldane NNN phase orientation nu_ij (+1 for clockwise, -1 for counter-clockwise).
        """
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

    def _build_hopping_templates(self):
        """
        One-time, parameter-independent precompute of hopping connectivity.

        For every (bond_type, site_i, site_j, spin) we cache the
        (bra_idx, ket_idx, sign) arrays returned by
        fast_ops.build_orbital_pair_template. These never change across
        different delta/U/V/flux/phase values -- only the scalar t_eff
        prefactor multiplying `sign` changes -- so we build them once here
        and reuse them on every subsequent _build_hopping call.
        """
        basis_np = self.basis.basis_states
        self._hop_templates: Dict[Tuple[str, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for bond_type, bonds in (('nn', self.lattice.get_bonds('nn')),
                                  ('nnn', self.lattice.get_bonds('nnn'))):
            for site_i, site_j in bonds:
                for spin in range(SPINS):
                    orb_i = site_i * SPINS + spin
                    orb_j = site_j * SPINS + spin
                    self._hop_templates[(bond_type, site_i, site_j, spin)] = \
                        build_orbital_pair_template(basis_np, orb_i, orb_j)

    def _build_hopping(self, flux_x: float = 0.0, flux_y: float = 0.0) -> sparse.coo_matrix:
        """
        Build Hermitian hopping Hamiltonian from the precomputed connectivity
        templates. No loop over the Fock basis happens here -- only a loop
        over bonds (a small, fixed number), each contributing a vectorized
        scalar-times-array multiply.
        """
        rows, cols, data = [], [], []

        tbc_nn = self._get_tbc_phases(flux_x, flux_y, 'nn')
        tbc_nnn = self._get_tbc_phases(flux_x, flux_y, 'nnn')
        hald_phases = self._get_chiralities()

        for (bond_type, site_i, site_j, spin), (bra_idx, ket_idx, sign) in self._hop_templates.items():
            if bond_type == 'nn':
                t_eff = T1 * tbc_nn[(site_i, site_j)]
            else:
                t_eff = T2 * hald_phases[(site_i, site_j)] * tbc_nnn[(site_i, site_j)]

            vals = sign * t_eff

            # c†_i c_j term
            rows.append(ket_idx)
            cols.append(bra_idx)
            data.append(vals)

            # c†_j c_i term (Hermitian conjugate)
            rows.append(bra_idx)
            cols.append(ket_idx)
            data.append(sign * np.conj(t_eff))

        H_hop = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.fock_dim, self.fock_dim),
            dtype=DTYPE
        )
        return H_hop
    
    def _build_staggered_potential(self, delta: float) -> sparse.dia_matrix:
        """
        Build staggered potential H_stag = Δ Σ_i (-1)^i n_i.

        Vectorized: uses the precomputed per-orbital occupation masks
        (self._occ, shape (NUM_ORBITALS, fock_dim)) instead of looping over
        every basis state and every orbital in Python.
        """
        if delta == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        for site in range(NUM_SITES):
            sublattice = self.lattice.site_to_sublattice[site]
            sign = 1.0 if sublattice == 0 else -1.0
            n_site = self._occ[site * SPINS + 0].astype(DTYPE_REAL) + \
                     self._occ[site * SPINS + 1].astype(DTYPE_REAL)
            diag += sign * n_site
        return sparse.diags(delta * diag, dtype=DTYPE_REAL)
    
    def _build_hubbard(self, U: float) -> sparse.dia_matrix:
        """Build Hubbard interaction H_U = U Σ_i n_↑ n_↓ (vectorized)."""
        if U == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        for site in range(NUM_SITES):
            n_up = self._occ[site * SPINS + 0]
            n_dn = self._occ[site * SPINS + 1]
            diag += (n_up & n_dn).astype(DTYPE_REAL)
        return sparse.diags(U * diag, dtype=DTYPE_REAL)
    
    def _build_coulomb(self, V: float) -> sparse.dia_matrix:
        """Build nearest-neighbor density interaction H_V = V Σ_<ij> n_i n_j (vectorized)."""
        if V == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        nn_bonds = self.lattice.get_bonds('nn')
        for i, j in nn_bonds:
            n_i = self._occ[i * SPINS + 0].astype(DTYPE_REAL) + self._occ[i * SPINS + 1].astype(DTYPE_REAL)
            n_j = self._occ[j * SPINS + 0].astype(DTYPE_REAL) + self._occ[j * SPINS + 1].astype(DTYPE_REAL)
            diag += n_i * n_j
        return sparse.diags(V * diag, dtype=DTYPE_REAL)


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
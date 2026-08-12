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


class HaldaneHubbardHamiltonian:
    """
    Build and manage the Haldane-Hubbard Hamiltonian.
    """
    def __init__(self):
        self.lattice = get_lattice()
        self.basis = get_fock_basis()
        self.fock_dim = len(self.basis.basis_states)
        
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

    def _build_hopping(self, flux_x: float = 0.0, flux_y: float = 0.0) -> sparse.coo_matrix:
        """
        Build Hermitian hopping Hamiltonian using fast single-pass dual insertion.
        """
        rows, cols, data = [], [], []
        
        nn_bonds = self.lattice.get_bonds('nn')
        nnn_bonds = self.lattice.get_bonds('nnn')
        basis_states = self.basis.basis_states
        basis_dict = self.basis.basis_dict
        
        def process_bond(site_i: int, site_j: int, t_val: complex):
            for spin in range(SPINS):
                orb_i = site_i * SPINS + spin
                orb_j = site_j * SPINS + spin
                
                for state_bra_idx, state_bra in enumerate(basis_states):

                    if not BasisOperations.is_occupied(state_bra, orb_j):
                        continue
                    state_temp = BasisOperations.set_unoccupied(state_bra, orb_j)

                    if BasisOperations.is_occupied(state_temp, orb_i):
                        continue
                    state_ket = BasisOperations.set_occupied(state_temp, orb_i)
                    
                    state_ket_idx = basis_dict.get(state_ket)
                    if state_ket_idx is not None:
                        sign = BasisOperations.fermionic_sign(state_bra, orb_j, orb_i)
                        
                        # c†_i c_j term
                        rows.append(state_ket_idx)
                        cols.append(state_bra_idx)
                        data.append(sign * t_val)
                        
                        # c†_j c_i term (Hermitian conjugate)
                        rows.append(state_bra_idx)
                        cols.append(state_ket_idx)
                        data.append(sign * np.conj(t_val))

        # First-neighbor hoppings (t1)
        tbc_phases = self._get_tbc_phases(flux_x, flux_y, 'nn')
        for i, j in nn_bonds:
            t_eff = T1 * tbc_phases[(i, j)]
            process_bond(i, j, t_eff)
            
        # Second-neighbor hoppings (t2)
        hald_phases = self._get_chiralities()
        tbc_phases = self._get_tbc_phases(flux_x, flux_y, 'nnn')
        for i, j in nnn_bonds:
            t_eff = T2 * hald_phases[(i, j)] * tbc_phases[(i, j)]
            process_bond(i, j, t_eff)
        
        H_hop = sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(self.fock_dim, self.fock_dim),
            dtype=DTYPE
        )
        return H_hop
    
    def _build_staggered_potential(self, delta: float) -> sparse.dia_matrix:
        """
        Build staggered potential H_stag = Δ Σ_i (-1)^i n_i.
        """
        if delta == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        for state_idx, state in enumerate(self.basis.basis_states):
            energy = 0.0
            for orb in range(NUM_ORBITALS):
                if BasisOperations.is_occupied(state, orb):
                    site = orb // SPINS
                    sublattice = self.lattice.site_to_sublattice[site]
                    sign = 1.0 if sublattice == 0 else -1.0
                    energy += sign * delta
            diag[state_idx] = energy
        return sparse.diags(diag, dtype=DTYPE_REAL)
    
    def _build_hubbard(self, U: float) -> sparse.dia_matrix:
        """Build Hubbard interaction H_U = U Σ_i n_↑ n_↓."""
        if U == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        for state_idx, state in enumerate(self.basis.basis_states):
            for site in range(NUM_SITES):
                n_up = 1 if (state & (1 << (site * SPINS + 0))) else 0
                n_dn = 1 if (state & (1 << (site * SPINS + 1))) else 0
                diag[state_idx] += U * n_up * n_dn
        return sparse.diags(diag, dtype=DTYPE_REAL)
    
    def _build_coulomb(self, V: float) -> sparse.dia_matrix:
        """Build nearest-neighbor density interaction H_V = V Σ_<ij> n_i n_j."""
        if V == 0:
            return sparse.dia_matrix((self.fock_dim, self.fock_dim), dtype=DTYPE_REAL)
        diag = np.zeros(self.fock_dim, dtype=DTYPE_REAL)
        nn_bonds = self.lattice.get_bonds('nn')
        for state_idx, state in enumerate(self.basis.basis_states):
            for i, j in nn_bonds:
                n_i = sum(1 for spin in range(SPINS) if state & (1 << (i * SPINS + spin)))
                n_j = sum(1 for spin in range(SPINS) if state & (1 << (j * SPINS + spin)))
                diag[state_idx] += V * n_i * n_j
        return sparse.diags(diag, dtype=DTYPE_REAL)

    #def _find_state_index(self, state: int) -> Optional[int]:
    #    return self.basis.basis_dict.get(state, None)


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
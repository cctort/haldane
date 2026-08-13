"""
Vectorized numpy helpers for Fock-basis operations.

The key idea used throughout this module: for a fixed pair of orbitals
(orb_i, orb_j), the *structure* of the single-particle hop c_i^dagger c_j
(which bra states it connects to which ket states, and what fermionic sign
each connection carries) depends only on the basis, not on any Hamiltonian
parameter (t, delta, U, V, flux, ...). So we compute that structure once,
vectorized over the whole basis with numpy, and cache it. Building the
actual Hamiltonian for a given set of parameters then only needs a handful
of scalar multiplies instead of a fresh O(fock_dim) Python loop.
"""

import numpy as np


def popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorized population count (number of set bits) for an int array."""
    if hasattr(np, "bitwise_count"):
        # numpy >= 2.0
        return np.bitwise_count(arr)
    # Fallback for numpy < 2.0
    arr = arr.copy()
    c = np.zeros_like(arr)
    while np.any(arr):
        c += arr & 1
        arr >>= 1
    return c


def build_orbital_pair_template(basis_states: np.ndarray, orb_i: int, orb_j: int):
    """
    Precompute the structural data for the single-particle term c_i^dagger c_j
    over the full basis: which bra indices connect to which ket indices, and
    the fermionic sign of each connection.

    Mirrors (and is numerically identical to) the following pure-Python loop:

        for bra_idx, state_bra in enumerate(basis_states):
            if not is_occupied(state_bra, orb_j):
                continue
            state_temp = set_unoccupied(state_bra, orb_j)
            if is_occupied(state_temp, orb_i):
                continue
            state_ket = set_occupied(state_temp, orb_i)
            ket_idx = basis_dict.get(state_ket)
            sign = fermionic_sign(state_bra, orb_j, orb_i)
            # connects bra_idx -> ket_idx with the given sign

    Requires basis_states to be SORTED ascending (used for vectorized
    lookup via np.searchsorted).

    Returns
    -------
    bra_idx : np.ndarray[int]   basis indices of contributing bra states
    ket_idx : np.ndarray[int]   basis indices of the corresponding ket states
    sign    : np.ndarray[int8]  fermionic sign of each connection
    """
    bit_i, bit_j = 1 << orb_i, 1 << orb_j

    occ_j = (basis_states & bit_j) != 0
    bra_idx_full = np.nonzero(occ_j)[0]
    bra_states_full = basis_states[occ_j]
    state_temp = bra_states_full & ~bit_j

    free_i = (state_temp & bit_i) == 0
    bra_idx = bra_idx_full[free_i]
    bra_states = bra_states_full[free_i]          # original state_bra (both conditions met)
    state_ket = state_temp[free_i] | bit_i

    ket_idx = np.searchsorted(basis_states, state_ket)
    # basis is fixed particle number, so state_ket is guaranteed present;
    # cheap paranoia check, remove if this ever becomes a hot path:
    # assert np.all(basis_states[ket_idx] == state_ket)

    lo, hi = (orb_i, orb_j) if orb_i < orb_j else (orb_j, orb_i)
    mask = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1)
    counts = popcount(bra_states & mask)
    sign = np.where(counts % 2 == 0, 1, -1).astype(np.int8)

    return bra_idx, ket_idx, sign


def build_occupation_masks(basis_states: np.ndarray, num_orbitals: int) -> np.ndarray:
    """
    Precompute, for every orbital, a boolean array over the whole basis
    indicating occupation. Shape: (num_orbitals, fock_dim).
    """
    fock_dim = basis_states.shape[0]
    occ = np.zeros((num_orbitals, fock_dim), dtype=bool)
    for orb in range(num_orbitals):
        occ[orb] = (basis_states & (1 << orb)) != 0
    return occ

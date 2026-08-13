"""
Sz-sector-factorized Fock basis construction.

Physics
-------
The Haldane-Hubbard Hamiltonian in this codebase never mixes spin: hopping
(t1, t2, the Haldane phase, and the twisted-boundary flux) is identical for
up and down and acts within a spin species only, and every interaction term
(U, V, staggered potential) is density-density. So besides total particle
number N, total Sz is *also* conserved, and fixing N_UP and N_DN separately
(config.N_UP / config.N_DN) restricts the calculation to one Sz sector
instead of diagonalizing across all of them at once.

Representation
--------------
Because the model doesn't mix spins, a many-body basis state can be built as
an independent pair (down-spin occupation pattern, up-spin occupation
pattern), each drawn from a *single-spin* sub-basis over NUM_SITES orbitals
(one orbital per site, no more spin-interleaving). We never form the flat
product basis explicitly: a combined state is identified by an
(dn_index, up_index) pair, and the many-body wavefunction is a
(dim_dn, dim_up) array -- see hamiltonian.py for how the Hamiltonian acts on
that shape directly (a "matrix-free" LinearOperator), so the huge product
dimension dim_dn * dim_up is never used to allocate an actual matrix.

This mirrors the I = 2^L * I_down + I_up convention (Lin & Gubernatis) that
factors the Hubbard-model Fock space along the same lines.
"""

import math
from itertools import combinations
import numpy as np
from .config import NUM_SITES, N_UP, N_DN, VERBOSE


class SpinBasis:
    """
    Fock basis for a single spin species: NUM_SITES orbitals (one per site),
    a fixed number of electrons occupied. Used independently for the up and
    down channels; if N_UP == N_DN the two channels can share one instance
    (see SzSectorBasis below).
    """

    def __init__(self, n_orbitals: int, n_electrons: int):
        self.num_orbitals = n_orbitals
        self.num_electrons = n_electrons
        self.states_np = self._construct()
        self.dim = self.states_np.shape[0]

    def _construct(self) -> np.ndarray:
        """
        Generate the C(n_orbitals, n_electrons) valid occupation patterns
        directly via itertools.combinations (rather than testing all
        2**n_orbitals candidates and filtering by popcount), and assemble
        them into integers with a single vectorized numpy reduction.
        """
        n, k = self.num_orbitals, self.num_electrons
        count = math.comb(n, k)
        if count == 0:
            return np.empty(0, dtype=np.int64)

        flat = np.fromiter(
            (idx for combo in combinations(range(n), k) for idx in combo),
            dtype=np.int64,
            count=count * k,
        )
        combo_arr = flat.reshape(count, k)
        states = (np.int64(1) << combo_arr).sum(axis=1)
        states.sort()
        return states

    def state_to_index(self, state: int) -> int:
        """Vectorized-lookup-friendly index search (no dict kept around)."""
        idx = int(np.searchsorted(self.states_np, state))
        if idx < self.dim and self.states_np[idx] == state:
            return idx
        return -1

    def index_to_state(self, index: int) -> int:
        return int(self.states_np[index])


class SzSectorBasis:
    """
    Combined (N_UP, N_DN) Sz-sector basis: a pair of SpinBasis instances plus
    the implicit combined dimension dim_dn * dim_up. A many-body index is
    (dn_index, up_index); the flat index (for anything that still wants one,
    e.g. warm-start vectors) is dn_index * dim_up + up_index, matching a
    row-major reshape to (dim_dn, dim_up).
    """

    def __init__(self):
        self.up = SpinBasis(NUM_SITES, N_UP)
        # Reuse the exact same SpinBasis object when the sectors are
        # identical (the common Sz=0, N_UP==N_DN case) -- avoids building
        # and holding two copies of what is the same array.
        self.dn = self.up if N_DN == N_UP else SpinBasis(NUM_SITES, N_DN)

        self.dim_up = self.up.dim
        self.dim_dn = self.dn.dim
        self.fock_dim = self.dim_up * self.dim_dn

        if VERBOSE:
            print("Constructed Sz-sector Fock basis:")
            print(f"  Orbitals per spin: {NUM_SITES}")
            print(f"  N_up={N_UP} (dim={self.dim_up}), N_dn={N_DN} (dim={self.dim_dn})")
            print(f"  Combined dimension: {self.fock_dim}"
                  f" (vs {math.comb(2*NUM_SITES, N_UP+N_DN)} for the unrestricted flat basis)")

    # -- convenience combined-index helpers -----------------------------
    def combined_index(self, dn_idx: int, up_idx: int) -> int:
        return dn_idx * self.dim_up + up_idx

    def split_index(self, flat_idx: int):
        dn_idx, up_idx = divmod(flat_idx, self.dim_up)
        return dn_idx, up_idx


_BASIS = None


def get_fock_basis() -> SzSectorBasis:
    """Get or create the global Sz-sector basis."""
    global _BASIS
    if _BASIS is None:
        _BASIS = SzSectorBasis()
    return _BASIS


def reset_basis():
    """Reset the global basis (for testing with different parameters)."""
    global _BASIS
    _BASIS = None

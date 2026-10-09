import numpy as np
from itertools import combinations
from .lattice import LATTICE
from .config import FILLING, SZ

def get_n_electrons(filling=FILLING, n_sites=LATTICE.n_sites):
    return np.round(2*filling*n_sites)

def build_states(n_electrons, n_sites):
    """
    Each configuration (a combination of n_electrons among n_sites elements)
    is represented as an integer using a bitstring: bit b = 1 means that
    site b is occupied, while bit b = 0 means that site b is empty.
    All integer representations are then stored in an array.

    Example:
        n_sites = 3, n_electrons = 2

        Possible configurations:
            sites (0, 1) -> 011 -> 3
            sites (0, 2) -> 101 -> 5
            sites (1, 2) -> 110 -> 6

        Returns:
            array([3, 5, 6])

    Args:
        n_sites: Total number of available sites.
        n_electrons: Number of electrons to place on the sites.

    Returns:
        A sorted NumPy array containing the integer representation
        of every possible configuration.
    """
    states = []

    for combo in combinations(range(n_sites), n_electrons):
        state = 0
        for b in combo:
            state += (1 << b) # << b shifts the bitstring left by b positions
        states.append(state)

    return np.array(sorted(states), dtype=np.int64)


class Basis:
    def __init__(self, n_electrons, n_sites=LATTICE.n_sites):
        self.states = build_states(n_electrons, n_sites)
        self.dim = len(self.states)

    def index(self, state):
        """For a given state, output the state index over the sorted states array"""
        i = int(np.searchsorted(self.states, state))
        return i if i < self.dim and self.states[i] == state else -1


class SpinfulBasis:
    def __init__(self, filling=FILLING, Sz=SZ, n_sites=LATTICE.n_sites):
        self.n_electrons = np.round(2*filling*n_sites)
        self.n_up = int(filling*n_sites + Sz)
        self.n_dn = int(filling*n_sites - Sz)
        
        self.up = Basis(self.n_up)
        # Reuse the same basis object when both sectors are identical (read-only objects)
        self.dn = self.up if self.n_dn == self.n_up else Basis(self.n_dn)
        self.dim_up = self.up.dim
        self.dim_dn = self.dn.dim
        self.dim = self.dim_up * self.dim_dn

BASIS = SpinfulBasis()
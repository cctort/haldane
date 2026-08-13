"""
Fixed particle number Fock basis construction.

A state is represented as a single integer (Fock state) where bit i indicates
occupancy of orbital i. We enumerate all states with Hamming weight = NUM_ELECTRONS.
"""

import math
from itertools import combinations
from typing import List
import numpy as np
from .config import NUM_ORBITALS, NUM_ELECTRONS, VERBOSE


class FockBasis:
    """
    Manages the fixed particle number Fock basis.
    
    Each state is represented as an integer where bit positions indicate
    orbital occupancies. For example, with 4 orbitals:
      state = 0101 means orbitals 0 and 2 are occupied
    """
    
    def __init__(self):
        """
        Initialize and construct the basis.
        """
        self.num_orbitals = NUM_ORBITALS
        self.num_electrons = NUM_ELECTRONS
        
        # List of all valid basis states
        self.basis_states: np.ndarray = np.empty(0, dtype=np.int64)
        
        self._construct_basis()
    
    def _construct_basis(self):
        """
        Enumerate all states with exactly num_electrons occupied orbitals.

        At half filling: NUM_ORBITALS = 24 and NUM_ELECTRONS = 12, thus a
        basis with C(24, 12) = 2704156 states.

        Implementation note: rather than testing all 2^num_orbitals
        candidate integers and filtering by popcount (16.7M candidates for
        the 24-orbital case, each requiring a bin(state).count('1') call),
        we generate the C(n, k) valid occupation patterns directly via
        itertools.combinations and convert them to integers with a single
        vectorized numpy reduction. This touches only the states that are
        actually in the basis, and does the bit-assembly at C speed instead
        of once per state in the Python interpreter.
        """
        n, k = self.num_orbitals, self.num_electrons
        count = math.comb(n, k)

        if count == 0:
            if VERBOSE:
                print("Constructed Fock basis: 0 states (invalid n/k)")
            return

        # Flatten combinations into a (count, k) index array, then reduce
        # each row to an integer state via sum of 2**orbital_index.
        flat = np.fromiter(
            (idx for combo in combinations(range(n), k) for idx in combo),
            dtype=np.int64,
            count=count * k,
        )
        combo_arr = flat.reshape(count, k)
        states = (np.int64(1) << combo_arr).sum(axis=1)
        states.sort()  # preserve the original ascending-integer ordering/index convention

        self.basis_states = states

        if VERBOSE:
            print(f"Constructed Fock basis:")
            print(f"  Orbitals: {self.num_orbitals}")
            print(f"  Electrons: {self.num_electrons}")
            print(f"  Basis dimension: {len(self.basis_states)}")
            print(f"  Expected (combinatorial): C({self.num_orbitals}, {self.num_electrons}) = {count}")
    
    def get_occupation_pattern(self, state: int) -> List[int]:
        """Get the list of orbital indices that are occupied"""
        occupied = []
        for orb in range(self.num_orbitals):
            if BasisOperations.is_occupied(state, orb):
                occupied.append(orb)
        return occupied
    
    def print_basis_sample(self, n_states: int = 10):
        """Print sample of basis states for debugging."""
        print(f"\nSample of first {min(n_states, len(self.basis_states))} basis states:")
        print(f"{'Index':<8} {'State (int)':<15} {'State (binary)':<30} {'Occupancy':<20}")
        print("-" * 75)
        
        for idx in range(min(n_states, len(self.basis_states))):
            state = self.basis_states[idx]
            occupied = self.get_occupation_pattern(state)
            binary = format(state, f'0{self.num_orbitals}b')
            occupied_str = f"[{','.join(map(str, occupied))}]"
            
            print(f"{idx:<8} {state:<15} {binary:<30} {occupied_str:<20}")


class BasisOperations:
    """Common operations on Fock basis states."""
    
    @staticmethod
    def count_particles(state: int) -> int:
        """Count number of occupied orbitals (Hamming weight)."""
        return bin(state).count('1')
    
    @staticmethod
    def is_occupied(state: int, orbital: int) -> bool:
        """Check if an orbital is occupied in a state."""
        return bool(state & (1 << orbital))
    
    @staticmethod
    def set_occupied(state: int, orbital: int) -> int:
        """Set an orbital to occupied."""
        return state | (1 << orbital)
    
    @staticmethod
    def set_unoccupied(state: int, orbital: int) -> int:
        """Set an orbital to unoccupied."""
        return state & ~(1 << orbital)
    
    @staticmethod
    def count_particles_between(state: int, orb1: int, orb2: int) -> int:
        """Count occupied orbitals between orb1 and orb2 (exclusive)."""
        min_orb = min(orb1, orb2)
        max_orb = max(orb1, orb2)
        
        count = 0
        for orb in range(min_orb + 1, max_orb):
            if state & (1 << orb):
                count += 1
        return count
    
    @staticmethod
    def fermionic_sign(state: int, orb1: int, orb2: int) -> int:
        count = BasisOperations.count_particles_between(state, orb1, orb2)
        return 1 if count % 2 == 0 else -1


_BASIS = None

def get_fock_basis() -> FockBasis:
    """Get or create the global Fock basis."""
    global _BASIS
    if _BASIS is None:
        _BASIS = FockBasis()
    return _BASIS

def reset_basis():
    """Reset the global basis (for testing with different parameters)."""
    global _BASIS
    _BASIS = None

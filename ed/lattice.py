"""
Honeycomb lattice construction for the 12A cluster.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Site:
    """Represent a lattice site."""
    idx: int
    sublattice: int  # A=0, B=1
    x: float
    y: float
    pos: np.ndarray
    
    def __repr__(self):
        return f"Site({self.idx}, sub={self.sublattice}, pos=({self.x:.3f},{self.y:.3f}))"


class HoneycombLattice:
    """
    12-site honeycomb cluster (12A geometry).
    
    Honeycomb lattice has two sublattices (A and B), each arranged in a triangular pattern.
    Nearest neighbors connect A-B atoms.
    Second-nearest neighbors connect within the same sublattice.
    """
    
    def __init__(self):
        """Initialize the 12-site honeycomb cluster."""
        self.num_sites = 12
        self.sites: Dict[int, Site] = {}
        self.nn_bonds: List[Tuple[int, int]] = []
        self.nnn_bonds: List[Tuple[int, int]] = []
        self.site_to_sublattice: Dict[int, int] = {}

        self.nn_bond_shift: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.nnn_bond_shift: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Primitive vectors for honeycomb lattice (a_nn = 1.0)
        self.a = 1.0
        
        # Supercell vectors for 12A cluster PBCs
        self.L1 = np.array([6*self.a, 0.])
        self.L2 = np.array([3*self.a/2, 3*np.sqrt(3)/2 * self.a])
        
        self._build_lattice()
        
    def _build_lattice(self):
        """Build the 12-site A cluster explicitly."""

        asqrt3 = np.sqrt(3) * self.a

        positions = [(0.0, 0.0), (2*self.a, 0.0), (3*self.a, 0.0), (5*self.a, 0.0),
                     (self.a/2, asqrt3/2), (3*self.a/2, asqrt3/2), (7*self.a/2, asqrt3/2), (9*self.a/2, asqrt3/2),
                     (2*self.a, asqrt3), (3*self.a, asqrt3), (5*self.a, asqrt3), (6*self.a, asqrt3)]

        sublattices = [0, 1, 0, 1,
                       1, 0, 1, 0,
                       1, 0, 1, 0,]

        for idx, ((x, y), sublattice) in enumerate(
            zip(positions, sublattices)
        ):
            self.sites[idx] = Site(
                idx=idx,
                sublattice=sublattice,
                x=x,
                y=y,
                pos=np.array([x, y])
            )

            self.site_to_sublattice[idx] = sublattice

        self._find_neighbors()

    def _get_distance(
            self, pos_i: np.ndarray, pos_j: np.ndarray
        ) -> Tuple[float, Tuple[int, int]]:
        """
        Calculate distance between sites and the shift (n1, n2)
        that minimizes |pos_j + n1*L1 + n2*L2 - pos_i|.
        """
        dr = pos_j - pos_i
        dist = np.linalg.norm(dr)
        shift = (0, 0)

        for n1 in [-1, 0, 1]:
            for n2 in [-1, 0, 1]:
                shift_vec = n1 * self.L1 + n2 * self.L2
                curr_dist = np.linalg.norm(dr + shift_vec)
                if curr_dist < dist - 1e-9:
                    dist = curr_dist
                    shift = (n1, n2)

        return dist, shift

    def _find_neighbors(self, nn_tol: float = 0.1, nnn_tol: float = 0.1):
        """
        Find nearest-neighbor and next-nearest-neighbor bonds, both via the
        self._get_distance).
        """
        
        nn_dist = self.a
        nnn_dist = self.a*np.sqrt(3)

        self.nn_bonds = []
        self.nnn_bonds = []
        self.nn_bond_shift = {}
        self.nnn_bond_shift = {}

        for i in range(self.num_sites):
            pos_i = np.array([self.sites[i].x, self.sites[i].y])
            for j in range(i + 1, self.num_sites):
                pos_j = np.array([self.sites[j].x, self.sites[j].y])

                dist, shift = self._get_distance(pos_i, pos_j)

                # NN bonds between A-B
                if abs(dist - nn_dist) < nn_tol and self.sites[i].sublattice != self.sites[j].sublattice:
                    self.nn_bonds.append((i, j))
                    self.nn_bond_shift[(i, j)] = shift

                # NNN bonds between A-A or B-B
                elif abs(dist - nnn_dist) < nnn_tol and self.sites[i].sublattice == self.sites[j].sublattice:
                    self.nnn_bonds.append((i, j))
                    self.nnn_bond_shift[(i, j)] = shift
    
    def get_bonds(self, bond_type : str) -> List[Tuple[int, int]]:
        """Get all bonds of type bond_type as (i, j) pairs with i < j."""
        bonds = self.nn_bonds if bond_type == 'nn' else self.nnn_bonds
        return bonds

    def get_bond_shift(self, bond_type: str) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Get all bond shifts of type bond_type as (i, j) pairs with i < j."""
        shifts = self.nn_bond_shift if bond_type == 'nn' else self.nnn_bond_shift
        return shifts
    r'''
    
    def get_neighbors(self, site_idx: int, shell: int = 1) -> List[int]:
        """
        Get neighbors of a site.
        
        Args:
            site_idx: Site index (0-11)
            shell: 1 for NN, 2 for NNN
            
        Returns:
            List of neighbor site indices
        """
        neighbors = set()
        
        bonds = self.nn_bonds if shell == 1 else (self.nnn_bonds if shell == 2 else [])
        for i, j in bonds:
            if i == site_idx:
                neighbors.add(j)
            elif j == site_idx:
                neighbors.add(i)
        
        return sorted(list(neighbors))
    '''
    
    def print_structure(self):
        """Print lattice structure for debugging."""
        print("\n" + "="*70)
        print("HONEYCOMB LATTICE STRUCTURE (12A Cluster)")
        print("="*70)
        print(f"{'Site':<6} {'Sublat':<8} {'Position':<20}")
        print("-"*70)
        for idx in range(self.num_sites):
            site = self.sites[idx]
            sub_label = 'A' if site.sublattice == 0 else 'B'
            pos_str = f"({site.x:6.3f}, {site.y:6.3f})"
            print(f"{idx:<6} {sub_label:<8} {pos_str:<20}")
        
        print(f"\nNearest neighbors ({len(self.nn_bonds)} bonds):")
        for i, j in self.nn_bonds:
            print(f"  {i} - {j}", end="  ")
        print()
        
        print(f"\nNext-nearest neighbors ({len(self.nnn_bonds)} bonds):")
        for i, j in self.nnn_bonds:
            print(f"  {i} - {j}", end="  ")
        print("\n" + "="*70)


_LATTICE = None

def get_lattice() -> HoneycombLattice:
    """Get or create the global lattice object."""
    global _LATTICE
    if _LATTICE is None:
        _LATTICE = HoneycombLattice()
    return _LATTICE

def reset_lattice():
    """Reset the global lattice (for testing)."""
    global _LATTICE
    _LATTICE = None
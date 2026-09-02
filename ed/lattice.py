import numpy as np

NUM_SITES = 12
a = 1.0
s3a = np.sqrt(3) * a

# 12-sites honeycomb cluster (12A geometry)
POSITIONS = np.array([
    (0.0,     0.0),
    (2*a,     0.0),
    (3*a,     0.0),
    (5*a,     0.0),
    (a/2,     s3a/2),
    (3*a/2,   s3a/2),
    (7*a/2,   s3a/2),
    (9*a/2,   s3a/2),
    (2*a,     s3a),
    (3*a,     s3a),
    (5*a,     s3a),
    (6*a,     s3a),
])

# Each site is either A=1 or B=-1
SUB_SIGN = np.array([1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1])

# Supercell vectors for periodic boundary conditions
L1 = np.array([6*a, 0.0])
L2 = np.array([1.5*a, 1.5*np.sqrt(3)*a])


def _min_dist(pos_i, pos_j):
    """Distance and integer shift (n1, n2) to the closest periodic copy."""
    dr = pos_i - pos_j
    best_d, best_shift = np.linalg.norm(dr), (0, 0)
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            d = np.linalg.norm(dr + n1 * L1 + n2 * L2)
            if d < best_d - 1e-9: # If two shifts give the same distance, keep the first
                best_d, best_shift = d, (n1, n2)
    return best_d, best_shift


class Lattice:
    """Nearest-neighbor (A-B and B-A, t1) and next-nearest-neighbor (A-A and B-B, t2)
    bonds, identified by their length and by the sublattices they connect."""

    def __init__(self):
        self.nn_bonds, self.nnn_bonds = [], [] # Lists of (i, j, (n1, n2)) tuples

        for i in range(NUM_SITES):
            for j in range(i + 1, NUM_SITES):
                dr = POSITIONS[i] - POSITIONS[j]
                
                # Check all relevant neighboring unit cells
                for n1 in (-1, 0, 1):
                    for n2 in (-1, 0, 1):
                        shift = (n1, n2)
                        d = np.linalg.norm(dr + n1 * L1 + n2 * L2)
                        
                        if abs(d - a) < 0.1 and SUB_SIGN[i] != SUB_SIGN[j]:
                            self.nn_bonds.append((i, j, shift))
                            
                        elif abs(d - a * np.sqrt(3)) < 0.1 and SUB_SIGN[i] == SUB_SIGN[j]:
                            self.nnn_bonds.append((i, j, shift))

    def chirality(self, i, j, shift):
        """Haldane phase sign nu_ij = +-1 for a next-nearest-neighbor bond."""
        n1, n2 = shift
        dx, dy = (POSITIONS[j] + n1 * L1 + n2 * L2) - POSITIONS[i] # coordinates of the bond vector from i to the periodic copy of j
        angle = np.arctan2(dy, dx) # angle of the bond vector measured from the x axis
        
        return SUB_SIGN[i] if np.sin(3 * angle) > 0 else -SUB_SIGN[i] # changes sign if bond goes counterclockwise around a hexagon

LATTICE = Lattice()
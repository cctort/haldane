import numpy as np
from collections import defaultdict
from .config import Config

def get_latt_sizes(cluster, n_max):
        if cluster == 'A':
            l1, l2 = np.array([3.0, 0.0]), np.array([-0.5, np.sqrt(3)/2]) # like 12 A
        elif cluster == 'C':
            l1, l2 = np.array([4.5, -np.sqrt(3)/2]), np.array([4.5, 3*np.sqrt(3)/2]) # like 12 C

        return l1*n_max[0], l2*n_max[1]

def get_latt_vecs(cell):
    if cell == 'honeycomb':
        a1, a2 = np.array([3.0, 0.0]), np.array([1.5, np.sqrt(3)/2])
        tau = np.array([[0.0, 0.0], [2.0, 0.0]])
        tau_sign = np.array([1, -1])

    return a1, a2, tau, tau_sign

class Lattice:
    """Nearest-neighbor (A-B and B-A, t1) and next-nearest-neighbor (A-A and B-B, t2)
    bonds, identified by their length and by the sublattices they connect."""

    def __init__(self, cfg: Config):

        self.cell = cfg.cell
        self.cluster = cfg.cluster
        self.n_max = cfg.n_max

        l1, l2 = get_latt_sizes(cfg.cluster, cfg.n_max)
        a1, a2, tau, tau_sign = get_latt_vecs(cfg.cell)

        M_inv = np.linalg.inv(np.array([l1, l2]).T)

        # Bound search region based on bounding box
        max_coord = np.sum([l1, l2], axis=0)
        n1_max = int(np.ceil(max_coord[0] / np.linalg.norm(a1))) + 2
        n2_max = int(np.ceil(max_coord[1] / np.linalg.norm(a2))) + 2

        positions = []
        signs = []
        for n1 in range(-n1_max, n1_max + 1):
            for n2 in range(-n2_max, n2_max + 1):
                for origin, sign in zip(tau, tau_sign):
                    pos = n1 * a1 + n2 * a2 + origin

                    # Pos coordinates in the LATT_LENGTHS basis
                    c1, c2 = M_inv @ pos

                    if (-1e-5 <= c1 < 1.0 - 1e-5 and -1e-5 <= c2 < 1.0 - 1e-5):
                        positions.append(pos)
                        signs.append(sign)

        self.positions = np.array(positions)
        self.signs = np.array(signs)
        self.n_sites = len(self.signs)

        bonds = []
        distances = []
        for i in range(self.n_sites):
            for j in range(i, self.n_sites):
                dr = self.positions[i] - self.positions[j]

                # Check all 9 shifts to find the shortest distance
                min_d = np.linalg.norm(max_coord)
                for n1 in (-1, 0, 1):
                    for n2 in (-1, 0, 1):
                        if i == j and n1 == 0 and n2 == 0:
                            continue
                        if i == j and (n1 < 0 or (n1 == 0 and n2 < 0)):
                            continue

                        vec = dr + n1 * l1 + n2 * l2
                        d = round(np.linalg.norm(vec), 5)
                        if d < min_d:
                            min_d = d

                # Store only the shifts that achieve this minimum distance
                for n1 in (-1, 0, 1):
                    for n2 in (-1, 0, 1):
                        if i == j and n1 == 0 and n2 == 0:
                            continue
                        if i == j and (n1 < 0 or (n1 == 0 and n2 < 0)):
                            continue

                        vec = dr + n1 * l1 + n2 * l2
                        d = round(np.linalg.norm(vec), 5)
                        if d == min_d:
                            bonds.append((i, j, (n1, n2), d))
                            distances.append(d)

        # Each bond degree (NN, NNN, ...) stores a list of (i, j, (n1, n2)) tuples
        self.bonds = defaultdict(list)
        sorted_dist = sorted(np.unique(distances))
        for i, j, shift, d in bonds:
            degree = sorted_dist.index(d) + 1
            self.bonds[degree].append((i, j, shift))


    def chirality(self, i, j, shift):
        """Haldane phase sign nu_ij = +-1 for a next-nearest-neighbor bond."""
        n1, n2 = shift
        l1, l2 = get_latt_sizes(self.cluster, self.n_max)
        dx, dy = (self.positions[i] + n1 * l1 + n2 * l2) - self.positions[j] # coordinates of the bond vector from i to the periodic copy of j
        angle = np.arctan2(dy, dx) # angle of the bond vector measured from the x axis
        
        return self.signs[i] if np.sin(3 * angle) > 0 else -self.signs[i] # changes sign if bond goes counterclockwise around a hexagon
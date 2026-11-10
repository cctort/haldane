import numpy as np

# In VMC there is no Fock basis to enumerate. A state is a *configuration*:
# for each spin species, the sites r_1 ... r_N occupied by electrons 1 ... N.


def random_positions(n_walkers, n_electrons, n_sites, rng):
    """(n_walkers, n_electrons) array of distinct random sites."""
    return np.argsort(rng.random((n_walkers, n_sites)), axis=1)[:, :n_electrons]


class Walkers:
    """
    Configurations of the W walkers living on this MPI rank.
    pos[s] : (W, N_s) site of each electron of spin s
    inv[s] : (W, N_s, N_s) inverse Slater matrix (set by the wavefunction)
    n_sites, n_el : number of sites and (N_up, N_dn), so other modules need no global lattice
    """

    def __init__(self, pos_up, pos_dn, n_sites):
        self.pos = [pos_up, pos_dn]
        self.W = len(pos_up)
        self.n_sites = n_sites
        self.n_el = (pos_up.shape[1], pos_dn.shape[1])
        self.inv = [None, None]

    def occ(self, s):
        """(W, N_sites) occupation numbers of spin s."""
        occ = np.zeros((self.W, self.n_sites))
        occ[np.arange(self.W)[:, None], self.pos[s]] = 1.0
        return occ

    def n_tot(self):
        return self.occ(0) + self.occ(1)
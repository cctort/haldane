import numpy as np


def hopping(lattice, cfg, flux_x=0.0, flux_y=0.0, t2=None, phi=None):
    """One-body matrix h with H_hop = sum_ij h[i,j] c_i^dag c_j.
    t2, phi default to the values in cfg."""
    t2 = cfg.t2 if t2 is None else t2
    phi = cfg.phi if phi is None else phi
    h = np.zeros((lattice.n_sites, lattice.n_sites), dtype=complex)

    for i, j, (n1, n2) in lattice.bonds[1]:
        amp = -cfg.t1 * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        h[i, j] += amp
        h[j, i] += np.conj(amp)

    for i, j, (n1, n2) in lattice.bonds[2]:
        nu = lattice.chirality(i, j, (n1, n2))
        amp = -t2 * np.exp(1j * np.pi * nu * phi) * np.exp(1j * (n1 * flux_x + n2 * flux_y))
        h[i, j] += amp
        h[j, i] += np.conj(amp)

    return h


class Hamiltonian:
    """
    H = sum h[i,j] c_i^dag c_j + delta sum_i xi_i n_i + U sum_i n_i^up n_i^dn + V sum_<ij> n_i n_j
    Provides the local energy E_loc(x) = <x|H|psi> / psi(x) instead of a matrix.
    """

    def __init__(self, cfg, lattice, delta, U, V, flux_x=0.0, flux_y=0.0):
        self.U, self.V = U, V
        self.hoppings = hopping(lattice, cfg, flux_x, flux_y)
        self.on_site = delta * lattice.signs
        self.Vmat = np.zeros((lattice.n_sites, lattice.n_sites))
        for i, j, _ in lattice.bonds[1]:
            self.Vmat[i, j] += 1.0

        # h is sparse: for every site store its (at most z) hopping partners and amplitudes,
        # padded with zero amplitude, so that E_loc only needs the ratios of these hops.
        partners = [np.nonzero(row)[0] for row in self.hoppings]
        z = max(len(p) for p in partners)
        self.nbr = np.zeros((lattice.n_sites, z), dtype=int)
        self.h_nbr = np.zeros((lattice.n_sites, z), dtype=complex)
        for i, p in enumerate(partners):
            self.nbr[i, :len(p)] = p
            self.h_nbr[i, :len(p)] = self.hoppings[i, p]

    def local_energy(self, wf, walkers):
        """
        E_loc for every walker, (W,) complex. In first quantization the fermionic sign
        of a hop cancels against the sign of the reordering of the determinant rows:
            E_hop = sum_a sum_j h[r_a, j] * psi(r_a -> j) / psi(r)
        """
        n_up, n_dn = walkers.occ(0), walkers.occ(1)
        n = n_up + n_dn
        E = n @ self.on_site
        E = E + self.U * (n_up * n_dn).sum(axis=1)
        E = E + self.V * ((n @ self.Vmat) * n).sum(axis=1)
        E = E.astype(complex)
        for s in (0, 1):
            pos = walkers.pos[s]                                            # (W, N_s)
            a = np.tile(np.arange(pos.shape[1]), (walkers.W, 1))            # every electron ...
            for k in range(self.nbr.shape[1]):                              # ... hopping to its k-th partner
                E += (self.h_nbr[pos, k] * wf.ratios(walkers, s, a, self.nbr[pos, k])).sum(axis=1)
        return E
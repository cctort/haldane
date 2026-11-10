import numpy as np

# All estimators take the current (local) walkers and return one value per walker.
# Average them with parallel.mean_all to pool the walkers of all MPI ranks.


def cdw(lattice, walkers):
    """Staggered charge structure factor (1/N) (sum_i xi_i n_i)^2."""
    return (walkers.n_tot() @ lattice.signs) ** 2 / lattice.n_sites


def sdw(lattice, walkers):
    """Staggered spin structure factor, same with n_up - n_dn."""
    return ((walkers.occ(0) - walkers.occ(1)) @ lattice.signs) ** 2 / lattice.n_sites


def double_occupancy(walkers):
    return (walkers.occ(0) * walkers.occ(1)).sum(axis=1) / walkers.n_sites


def density_matrix(wf, walkers):
    """
    (rho_up, rho_dn), each (W, N_sites, N_sites): the contribution of every walker to
    rho[j,i] = <c_j^dag c_i> (same convention as ED). The electron sitting on j hops to i,
    weighted by the amplitude ratio. Average over walkers with parallel.mean_all.
    """
    n_sites, rows = walkers.n_sites, np.arange(walkers.W)
    targets = np.tile(np.arange(n_sites), (walkers.W, 1))                    # (W, N_sites): every site i
    out = []
    for s in (0, 1):
        rho = np.zeros((walkers.W, n_sites, n_sites), dtype=complex)
        for a in range(walkers.n_el[s]):
            ratio = wf.ratios(walkers, s, np.full(targets.shape, a), targets)   # electron a hops to every i
            rho[rows, walkers.pos[s][:, a], :] += ratio                         # row j = r_a
        out.append(rho)
    return out
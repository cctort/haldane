from dataclasses import dataclass


@dataclass
class Config:
    """All parameters in one place. Create with Config() or Config(t2=0.1, n_max=(3, 3), ...)."""

    # Lattice (see lattice.get_latt_sizes / get_latt_vecs). n_max=(2, 3), cluster 'A' = old 12 A
    cell: str = 'honeycomb'
    cluster: str = 'A'
    n_max: tuple = (2, 3)

    # Occupation: None = half filling (one electron per site), else the total number
    num_electrons: int = None

    # Hoppings
    t1: float = 1.0
    t2: float = 0.2
    phi: float = 0.5          # in units of pi

    # Monte Carlo sampling
    seed: int = 1234
    n_walkers: int = 512      # independent Markov chains, TOTAL over all MPI ranks
    n_therm: int = 200        # sweeps of initial thermalization
    n_skip: int = 2           # sweeps between two measurements
    n_meas: int = 20          # measurements per walker per optimization iteration

    # Stochastic reconfiguration
    n_iter: int = 100
    sr_lr: float = 0.05
    sr_eps: float = 1e-3      # diagonal shift of the S matrix
    sr_max_step: float = 0.1  # max |delta theta_k| per iteration

    verbose: bool = False

    def n_electrons(self, n_sites):
        """(N_up, N_dn)."""
        total = n_sites if self.num_electrons is None else self.num_electrons
        return total // 2, total - total // 2
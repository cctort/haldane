from . import parallel
import numpy as np
from . import observables as obs

def mc_step(wf, walkers, rng):
    """
    One Metropolis step: per spin species and walker, move a random electron to a random
    site without a same-spin electron (symmetric proposal); accept with min(1, |psi'/psi|^2).
    Returns the number of accepted moves.
    """
    W = walkers.W
    n_acc = 0
    for s in (0, 1):
        a = rng.integers(0, walkers.n_el[s], W)
        scores = rng.random((W, walkers.n_sites))
        scores[walkers.occ(s) > 0] = 2.0
        j = scores.argmin(axis=1)
        ratio = wf.ratios(walkers, s, a[:, None], j[:, None])[:, 0]
        acc = np.nonzero(rng.random(W) < np.abs(ratio) ** 2)[0]
        wf.move(walkers, s, acc, a[acc], j[acc])
        n_acc += len(acc)
    return n_acc


def run_sweeps(wf, walkers, rng, n_sweeps):
    """Returns the acceptance rate (of the local walkers)."""
    n_steps = max(walkers.n_el)
    n_acc = 0
    for _ in range(n_sweeps):
        n_acc += sum(mc_step(wf, walkers, rng) for _ in range(n_steps))
        wf.update_inverse(walkers)    # fresh inverses: removes the drift of the rank-1 updates
    return n_acc / (2 * walkers.W * n_sweeps * n_steps)


def collect(wf, ham, walkers, rng, n_meas, n_skip):
    """
    n_meas measurements per local walker, n_skip sweeps apart:
    E_loc (n_meas, W), O (n_meas, W, P), acceptance rate.
    """
    E, O, acc = [], [], 0.0
    for _ in range(n_meas):
        acc += run_sweeps(wf, walkers, rng, n_skip)
        E.append(ham.local_energy(wf, walkers))
        O.append(wf.log_derivatives(walkers))
    return np.array(E), np.array(O), acc / n_meas

def default_observables(wf, ham):
    """name -> function(walkers) returning one value per walker (first axis = walker)."""
    lat = wf.lattice
    return {
        'energy': lambda w: ham.local_energy(wf, w).real / lat.n_sites,
        'cdw': lambda w: obs.cdw(lat, w),
        'sdw': lambda w: obs.sdw(lat, w),
        'docc': lambda w: obs.double_occupancy(w),
        'rho': lambda w: np.stack(obs.density_matrix(wf, w), axis=1),    # (W, 2, N_sites, N_sites): up, dn
    }


def measure(cfg, wf, ham, n_meas=200, observables=None):
    """
    Samples |psi|^2 with fresh walkers (thermalization, then n_meas measurements n_skip sweeps apart) and
    returns {name: (mean, error)}. The error is the spread of the walker averages: the walkers are
    independent chains, also across the MPI ranks. Pass a subset of default_observables(wf, ham), or your
    own functions, as `observables` to measure less (the density matrix is the expensive one).
    """
    observables = default_observables(wf, ham) if observables is None else observables
    rng = np.random.default_rng([cfg.seed, 1, parallel.RANK])     # not the stream used by optimize
    walkers = wf.init_walkers(parallel.n_local(cfg.n_walkers), rng)
    run_sweeps(wf, walkers, rng, cfg.n_therm)

    total = {name: 0 for name in observables}
    for _ in range(n_meas):
        run_sweeps(wf, walkers, rng, cfg.n_skip)
        for name, f in observables.items():
            total[name] = total[name] + f(walkers)                # running sum per walker

    result = {}
    for name, t in total.items():
        chains = parallel.concat_all(t / n_meas)                  # (W_total, ...): one value per chain
        result[name] = chains.mean(axis=0), chains.std(axis=0, ddof=1) / np.sqrt(len(chains))
    return result
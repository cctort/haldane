from . import parallel
import numpy as np
from .sampler import run_sweeps, collect


def sr_step(E, O):
    """
    E: (N,) complex local energies, O: (N, P) complex log-derivatives, N = local samples.
    Returns S, f, the pooled mean energy and its variance, for the SR equation
        (S + eps) dtheta = -lr * f  with
        S_kl = Re <(O_k - <O_k>)^* (O_l - <O_l>)>,    f_k = Re <(O_k - <O_k>)^* (E - <E>)>
    """
    N = parallel.sum_all(len(E))
    E_mean = parallel.sum_all(E.sum()) / N
    dE = E - E_mean
    dO = O - parallel.sum_all(O.sum(axis=0)) / N                  # (N, P)

    S = parallel.sum_all(dO.conj().T @ dO).real / N               # (P, P)
    f = parallel.sum_all(dO.conj().T @ dE).real / N               # (P,)
    var = parallel.sum_all((np.abs(dE) ** 2).sum()) / N
    return S, f, E_mean, var


def optimize(cfg, wf, ham):
    """
    Stochastic reconfiguration. Returns a dict with one entry per iteration:
        e_loc  (n_iter,)    energy per site
        var    (n_iter,)    variance of the local energy
        mf_gap (n_iter, 2)  mean-field gap [up, dn]
        acc    (n_iter,)    acceptance rate (local walkers of this rank)
        and one array per free parameter key of wf: t2, phi, delta, m (n_iter,); js_ch, js_sp (n_iter, wf.n_js[key]).
    Every entry belongs to the parameters that produced that iteration's energy.
    """
    rng = np.random.default_rng([cfg.seed, parallel.RANK])        # independent stream per rank
    walkers = wf.init_walkers(parallel.n_local(cfg.n_walkers), rng)
    run_sweeps(wf, walkers, rng, cfg.n_therm)

    P = wf.n_params
    eloc, gaps, accs = [], [], []
    params = {key: [] for key in wf.free_keys}
    for it in range(cfg.n_iter):
        E, O, acc = collect(wf, ham, walkers, rng, cfg.n_meas, cfg.n_skip)   # (n_meas, W), (n_meas, W, P)
        S, f, E_mean, var = sr_step(E.ravel(), O.reshape(-1, P))

        step = -cfg.sr_lr * np.linalg.solve(S + cfg.sr_eps * np.eye(P), f)
        step = np.clip(step, -cfg.sr_max_step, cfg.sr_max_step)

        for key in wf.free_keys:
            params[key].append(wf.par[key])
        gaps.append(wf.gap)                                       # (2,) mean-field gap of up and down
        eloc.append(E_mean.real / wf.lattice.n_sites)
        accs.append(acc)

        theta = wf.get_params()
        while True:                                               # halve the step if the Fermi level becomes degenerate
            try:
                wf.set_params(theta + step)
                break
            except ValueError:
                step = step / 2
        wf.update_inverse(walkers)                                # Phi changed, so inv is stale

    data = {'e_loc': np.array(eloc), 'mf_gap': np.array(gaps), 'acc': np.array(accs)}
    params = {key: np.array(vals) for key, vals in params.items()}   # scalars: (n_iter,), js_ch/js_sp: (n_iter, n_js)
    return {**data, **params}
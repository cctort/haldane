import numpy as np
from .basis import Walkers, random_positions
from .hamiltonian import hopping

MF_KEYS = ('t2', 'phi', 'delta', 'm')      # mean-field parameters (scalars)
JS_KEYS = ('js_ch', 'js_sp')               # Jastrow parameters (arrays): charge and spin, one per pair class
KEYS = MF_KEYS + JS_KEYS


def degree_matrix(lattice):
    """degree[i, j] = bond degree of the pair (i, j): 0 = on-site, 1 = NN, 2 = NNN, ..."""
    degree = np.zeros((lattice.n_sites,) * 2, dtype=int)
    for d, blist in lattice.bonds.items():
        for i, j, _ in blist:
            degree[i, j] = degree[j, i] = d
    return degree


def pair_classes(lattice, kind):
    """
    cls[i, j] = index of the Jastrow parameter of the pair of sites (i, j), symmetric.
        'degree'     : one parameter per bond degree (distance)
        'bond'       : as 'degree' for degree >= 1, plus one Gutzwiller (on-site) parameter per site, which is
                       not translation invariant. Parameters: n_sites on-site ones first, then degrees 1, 2, ...
    """
    n = lattice.n_sites
    if kind == 'degree':
        key = degree_matrix(lattice)
    elif kind == 'bond':
        key = degree_matrix(lattice) + n                   # degrees >= 1 become n + degree ...
        key[np.arange(n), np.arange(n)] = np.arange(n)     # ... and every site gets its own on-site class
    else:
        raise ValueError(f"Unknown Jastrow kind '{kind}'; valid: 'degree', 'bond'")
    return np.unique(key, return_inverse=True)[1].reshape(n, n)       # relabel as 0, 1, 2, ...


def jastrow_change(v, n, sigma, w, i, j):
    """Change of -1/2 n.v.n when an electron of sign sigma (1 for charge) moves from site i to j; w: walker index."""
    vn = n @ v
    return sigma * (vn[w, i] - vn[w, j]) - 0.5 * (v[i, i] + v[j, j]) + v[i, j]


class SlaterJastrow:
    """
    psi(x) = det(up) det(dn) exp(J),   J = -1/2 n.v_ch.n - 1/2 s.v_sp.s,   n = n_up + n_dn,  s = n_up - n_dn.
    jastrow: how the entries of v_ch, v_sp are tied to the free parameters js_ch, js_sp (see pair_classes);
    one string for both, or a dict like {'js_ch': 'degree', 'js_sp': 'bond'} (missing keys: 'degree').
    The parameter of pair class c is wf.par[key][c]; the class of the pair (i, j) is wf.cls[key][i, j].
    """

    def __init__(self, cfg, lattice, free=None, fixed=None, flux_x=0.0, flux_y=0.0, jastrow='degree'):
        free, fixed = dict(free or {}), dict(fixed or {})
        if set(free) & set(fixed):
            raise ValueError(f'Parameters both free and fixed: {set(free) & set(fixed)}')
        if (set(free) | set(fixed)) - set(KEYS):
            raise ValueError(f'Unknown parameters {(set(free) | set(fixed)) - set(KEYS)}; valid keys: {KEYS}')

        self.cfg, self.lattice = cfg, lattice
        self.n_el = cfg.n_electrons(lattice.n_sites)
        self.flux = (flux_x, flux_y)
        if isinstance(jastrow, str):
            jastrow = dict.fromkeys(JS_KEYS, jastrow)
        if set(jastrow) - set(JS_KEYS):
            raise ValueError(f'Unknown jastrow keys {set(jastrow) - set(JS_KEYS)}; valid keys: {JS_KEYS}')
        self.cls = {k: pair_classes(lattice, jastrow.get(k, 'degree')) for k in JS_KEYS}
        self.n_js = {k: int(self.cls[k].max()) + 1 for k in JS_KEYS}

        default = {'t2': cfg.t2, 'phi': cfg.phi, 'delta': 0.0, 'm': 0.0, 'js_ch': 0.0, 'js_sp': 0.0}
        params = {**default, **fixed, **free}
        for k in JS_KEYS:
            params[k] = self.js_params(k, params[k])

        self.free_keys = [k for k in KEYS if k in free]
        self.fixed = {k: params[k] for k in KEYS if k not in free}
        self.theta = np.concatenate([np.atleast_1d(np.asarray(params[k], float)) for k in self.free_keys]
                                    + [np.zeros(0)])
        self.n_params = len(self.theta)
        self.update()

    def js_params(self, k, v):
        """0 -> zeros, 'rnd' -> small random numbers (same seed on every rank), array -> itself."""
        if isinstance(v, str):
            if v != 'rnd':
                raise ValueError(f"{k} must be 0, 'rnd' or an array")
            return 0.1 * np.random.default_rng(self.cfg.seed + JS_KEYS.index(k)).standard_normal(self.n_js[k])
        return np.broadcast_to(np.asarray(v, float), (self.n_js[k],)).copy()

    def get_params(self):
        return self.theta.copy()

    def set_params(self, theta):
        self.theta = np.array(theta, dtype=float)
        self.update()

    def unpack(self):
        """theta -> dict with the value of every parameter (free from theta, fixed as stored)."""
        par, n = dict(self.fixed), 0
        for k in self.free_keys:
            if k in JS_KEYS:
                par[k], n = self.theta[n:n + self.n_js[k]], n + self.n_js[k]
            else:
                par[k], n = self.theta[n], n + 1
        return par

    def update(self):
        """Diagonalize the mean-field Hamiltonian; store orbitals and their derivatives."""
        self.par = par = self.unpack()
        t2, phi, delta, m = (par[k] for k in MF_KEYS)
        n_sites = self.lattice.n_sites
        hop = lambda t, p: hopping(self.lattice, self.cfg, *self.flux, t, p)
        xi = np.diag(self.lattice.signs.astype(float))
        step = 1e-6
        dh = [(hop(t2 + step, phi) - hop(t2 - step, phi)) / (2 * step), # dh/dt2
              (hop(t2, phi + step) - hop(t2, phi - step)) / (2 * step)] # dh/dphi
        mf_idx = [MF_KEYS.index(k) for k in self.free_keys if k in MF_KEYS]

        self.v_ch, self.v_sp = (par[k][self.cls[k]] for k in JS_KEYS) # (n_sites, n_sites)
        self.Phi, self.dPhi, self.gap = [], [], []
        for s, sigma in enumerate((1.0, -1.0)):
            N = self.n_el[s]
            energy, vec = np.linalg.eigh(hop(t2, phi) + (delta + sigma * m) * xi)
            self.gap.append(energy[N] - energy[N - 1]) # mean-field gap of spin s
            if self.gap[-1] < 1e-8:
                raise ValueError(f'Mean-field Fermi level is degenerate: gap = {self.gap[-1]} for spin {s}')
            occup, empty = vec[:, :N], vec[:, N:]
            
            # first-order perturbation theory: occupied orbitals mix with unoccupied ones
            dH = dh + [xi, sigma * xi] # d h_MF / d(t2, phi, delta, m)
            denom = energy[:N][None, :] - energy[N:][:, None] # (N_sites - N, N)
            dPhi = [empty @ ((empty.conj().T @ dH[k] @ occup) / denom) for k in mf_idx]
            self.Phi.append(occup)
            self.dPhi.append(np.array(dPhi).reshape(len(mf_idx), n_sites, N)) # (n_free_orb, N_sites, N)

    def update_inverse(self, walkers):
        """Recompute the inverse Slater matrices of all walkers from scratch."""
        for s in (0, 1):
            walkers.inv[s] = np.linalg.inv(self.Phi[s][walkers.pos[s]])

    def init_walkers(self, n_walkers, rng):
        n_sites = self.lattice.n_sites
        walkers = Walkers(*[random_positions(n_walkers, self.n_el[s], n_sites, rng) for s in (0, 1)], n_sites)
        while True:                                  # redraw walkers with a vanishing amplitude
            bad = np.zeros(n_walkers, dtype=bool)
            for s in (0, 1):
                bad |= np.abs(np.linalg.det(self.Phi[s][walkers.pos[s]])) < 1e-12
            if not bad.any():
                break
            for s in (0, 1):
                walkers.pos[s][bad] = random_positions(bad.sum(), self.n_el[s], n_sites, rng)
        self.update_inverse(walkers)
        return walkers

    def move(self, walkers, s, rows, a, j):
        """
        Move electron a[k] of spin s to site j[k] in the walkers rows[k], and update their inverse
        Slater matrices with the Sherman-Morrison formula (O(N^2) instead of O(N^3)).
        Replacing row a of the Slater matrix A by u = Phi[j] gives
            A'^-1 = A^-1 - A^-1[:, a] (u A^-1 - e_a) / r,     r = u . A^-1[:, a]  (det ratio).
        """
        if len(rows) == 0:
            return
        k = np.arange(len(rows))
        inv = walkers.inv[s][rows]                                  # (R, N, N)
        u = self.Phi[s][j]                                          # (R, N) new row of A
        col = inv[k, :, a]                                          # (R, N) column a of A^-1
        r = (u * col).sum(axis=1)                                   # (R,)
        row = (u[:, None, :] @ inv)[:, 0, :]                        # (R, N) u A^-1
        row[k, a] -= 1.0
        walkers.inv[s][rows] = inv - col[:, :, None] * row[:, None, :] / r[:, None, None]
        walkers.pos[s][rows, a] = j

    def ratios(self, walkers, s, a, j):
        """
        psi(x')/psi(x) if electron a[w, k] of spin s hops to site j[w, k]; a, j: (W, K), result (W, K).
        """
        # determinant ratio (Cramer): orbitals at j against column a of the inverse Slater matrix.
        # It is 0 for a same-spin occupied j and 1 for j = r_a.
        w = np.arange(walkers.W)[:, None]                                        # walker index, (W, 1)
        cols = walkers.inv[s].transpose(0, 2, 1)[w, a]                           # (W, K, N): inv[w, :, a[w, k]]
        det = (cols * self.Phi[s][j]).sum(axis=2)                                # (W, K)

        # Jastrow ratio exp(J(x') - J(x)) for an electron of spin sigma moving from site i to j, per channel:
        #   dJ = sigma ((v m)_i - (v m)_j) - (v_ii + v_jj) / 2 + v_ij,   m = n (charge, sigma = 1) or s (spin)
        i = walkers.pos[s][w, a]                                                 # (W, K) current sites
        n_up, n_dn = walkers.occ(0), walkers.occ(1)
        dJ = jastrow_change(self.v_ch, n_up + n_dn, 1.0, w, i, j) \
           + jastrow_change(self.v_sp, n_up - n_dn, 1.0 - 2 * s, w, i, j)
        return det * np.exp(dJ)

    def log_derivatives(self, walkers):
        """O_k(x) = d ln psi(x) / d theta_k for the free parameters, (W, n_params) complex."""
        # Slater part: d ln det A = tr(A^-1 dA) = sum_ab dA[a, b] * inv[b, a]
        O_mf = 0
        for s in (0, 1):
            dA = self.dPhi[s][:, walkers.pos[s]]                                 # (n_free_orb, W, N_s, N_s)
            O_mf = O_mf + (dA * walkers.inv[s].transpose(0, 2, 1)).sum(axis=(2, 3))   # (n_free_orb, W)

        # Jastrow part: dJ/dv_c = -1/2 * sum over ordered electron pairs (k, l) of class c of sg_k sg_l,
        # with sg = 1 for the charge and sg = +-1 (spin of the electron) for the spin channel
        W = walkers.W
        r = np.concatenate(walkers.pos, axis=1)                                  # (W, N_tot)
        sigma = np.repeat([1.0, -1.0], self.n_el)                                # (N_tot,)
        O_jas = []
        for k, sg in (('js_ch', np.ones_like(sigma)), ('js_sp', sigma)):
            if k in self.free_keys:
                C = self.n_js[k]
                pair_class = self.cls[k][r[:, :, None], r[:, None, :]]           # (W, N_tot, N_tot)
                flat = pair_class + C * np.arange(W)[:, None, None]              # one histogram per walker
                weight = np.broadcast_to(sg[:, None] * sg[None, :], flat.shape)
                O_jas.append(-0.5 * np.bincount(flat.ravel(), weight.ravel(), minlength=W * C).reshape(W, C))

        return np.concatenate([O_mf.T] + O_jas, axis=1)
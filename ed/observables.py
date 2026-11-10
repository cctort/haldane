from fractions import Fraction

import numpy as np
from scipy import sparse

from .basis import SpinfulBasis
from .lattice import Lattice
from .hamiltonian import hop_element, occupation_table
from .config import Config
from .ed_solver import EDSolver


class Observables:
    def __init__(self, cfg: Config, basis: SpinfulBasis, lattice: Lattice, solver: EDSolver):
        self.solver = solver
        self.basis = basis
        self.lattice = lattice

        self.occ_up = occupation_table(basis.up.states, lattice.n_sites)
        self.occ_dn = self.occ_up if basis.dn is basis.up else occupation_table(basis.dn.states, lattice.n_sites)
        self.stag_up = lattice.signs @ self.occ_up
        self.stag_dn = lattice.signs @ self.occ_dn

        self.n_flux = cfg.n_flux
        self.flux_max_refine = cfg.flux_max_refine
        self.flux_max_angle = cfg.flux_max_angle      # refine where |plaquette flux| exceeds this

    def prob(self, psi):
        """|psi|^2 reshaped to (dim_dn, dim_up), normalized."""
        prob = np.abs(psi.reshape(self.basis.dim_dn, self.basis.dim_up)) ** 2
        return prob / prob.sum()

    def pair_terms(self, psi):
        """Site-resolved <n_i,up n_j,up>, <n_i,dn n_j,dn>, <n_i,up n_j,dn> (shared by charge and Sz correlations)."""
        prob = self.prob(psi)  # shape: (dim_dn, dim_up)

        p_up = prob.sum(axis=0)  # shape: (dim_up,)
        p_dn = prob.sum(axis=1)  # shape: (dim_dn,)

        n_up_n_up = (self.occ_up * p_up) @ self.occ_up.T
        n_dn_n_dn = (self.occ_dn * p_dn) @ self.occ_dn.T
        n_up_n_dn = self.occ_up @ prob.T @ self.occ_dn.T
        return n_up_n_up, n_dn_n_dn, n_up_n_dn

    def corr_ch(self, psi):
        """ Spatial charge correlation elements <n_i n_j> / N."""
        n_up_n_up, n_dn_n_dn, n_up_n_dn = self.pair_terms(psi)
        return (n_up_n_up + n_up_n_dn + n_up_n_dn.T + n_dn_n_dn) / self.lattice.n_sites

    def corr_sz(self, psi):
        """ Spatial spin correlation elements <Sz_i Sz_j> / N. """
        n_up_n_up, n_dn_n_dn, n_up_n_dn = self.pair_terms(psi)
        return (n_up_n_up - n_up_n_dn - n_up_n_dn.T + n_dn_n_dn) / (4 * self.lattice.n_sites)

    def hopping_op(self, states, i, j):
        r"""Build sparse operator matrix for c^\dagger_i c_j using hop_element."""
        dim = len(states)
        rows, cols, vals = [], [], []
        for ket, bra, sign in hop_element(states, i, j):
            rows.append(bra)
            cols.append(ket)
            vals.append(sign)
        return sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float).tocsr()

    def corr_sx(self, psi):
        """Spatial transverse spin correlation elements <Sx_i Sx_j> / N."""
        dim_dn, dim_up = self.basis.dim_dn, self.basis.dim_up
        M = psi.reshape(dim_dn, dim_up)
        n_sites = self.lattice.n_sites
        prob = self.prob(psi)
        n_up = self.occ_up @ prob.sum(axis=0)  # <n_{i,up}>
        n_dn = self.occ_dn @ prob.sum(axis=1)  # <n_{i,dn}>
        corr_sx_mat = np.zeros((n_sites, n_sites), dtype=float)

        for i in range(n_sites):
            for j in range(n_sites):
                # <S+_i S-_j> = <c^\dagger_{i,up} c_{i,dn} c^\dagger_{j,dn} c_{j,up}>
                #             = delta_ij <n_{i,up}> - <(c^\dagger_{i,up} c_{j,up}) (c^\dagger_{j,dn} c_{i,dn})>
                up_op = self.hopping_op(self.basis.up.states, i, j)
                dn_op = self.hopping_op(self.basis.dn.states, j, i)
                corr_sx_mat[i, j] = -np.vdot(M, dn_op @ M @ up_op.T).real
            corr_sx_mat[i, i] += (n_up[i] + n_dn[i]) / 2  # delta_ij term, averaged with <S-_i S+_i>

        # <Sx_i Sx_j> = (<S+_i S-_j> + <S-_i S+_j>) / 4 = Re <S+_i S-_j> / 2
        return corr_sx_mat / (2 * n_sites)

    def S_q0(self, corr):
        return corr.sum(axis=(-2, -1))

    def stagger(self, corr):
        signs = self.lattice.signs
        return corr * np.outer(signs, signs)

    def S_qpi(self, corr):
        stagg_corr = self.stagger(corr)
        return stagg_corr.sum(axis=(-2, -1))

    def density_matrix(self, psi):
        """
        rho_ij = <c_i^dagger c_j> per spin.
        """
        M = psi.reshape(self.basis.dim_dn, self.basis.dim_up)
        G = M.conj().T @ M     # (dim_up, dim_up)
        H2 = M @ M.conj().T    # (dim_dn, dim_dn)

        diagG, diagH2 = np.diag(G).real, np.diag(H2).real
        rho_up = np.diag(self.occ_up @ diagG).astype(complex)
        rho_dn = np.diag(self.occ_dn @ diagH2).astype(complex)

        for i in range(self.lattice.n_sites):
            for j in range(self.lattice.n_sites):
                if i == j:
                    continue
                for bra, ket, sign in hop_element(self.basis.up.states, j, i):
                    rho_up[j, i] += sign * G[ket, bra]
                for bra, ket, sign in hop_element(self.basis.dn.states, j, i):
                    rho_dn[j, i] += sign * H2[bra, ket]

        return rho_up, rho_dn

    def solve_twist(self, delta, U, V, phi_x, phi_y, v0, want_gap, site_dis=None, bond_dis=None):
        """Ground state at one twist: the unit of work handed to an MPI worker.
        Returns (psi, None), or (None, gap) when want_gap."""
        if want_gap:
            _, _, gap_min = self.solver.ground_state(delta, U, V, phi_x, phi_y, v0, True, site_dis, bond_dis)
            return None, gap_min
        _, psi = self.solver.ground_state(delta, U, V, phi_x, phi_y, v0, False, site_dis, bond_dis)
        return psi, None

    def chern_job(self, delta, U, V, site_dis=None, bond_dis=None, gap=True):
        """Adaptive FHS Chern number as a coroutine, so the caller decides where the solves run.

        Yields a list of independent requests (phi_x, phi_y, v0, want_gap) and expects the list of
        solve_twist results, same order, back through .send(). Returns (StopIteration.value) a dict:
        chern, curvature (on the adaptive grid phi_x x phi_y), gap (minimum gap) and gap_phi (its twist angles).
        Keeps no state on self, so any number of jobs can be interleaved on one Observables.
        """
        two_pi = 2 * np.pi
        xs = [Fraction(k, self.n_flux) for k in range(self.n_flux)]
        ys = list(xs)
        cache = {}  # (u_x, u_y) -> ground state

        def angle(u):
            return two_pi * (float(u) - 0.5)

        def midpoint(axis, i):
            nxt = axis[i + 1] if i + 1 < len(axis) else axis[0] + 1
            return (axis[i] + nxt) / 2

        def link(a, b):
            """Phase of <a|b>; arbitrary (1.0) if the overlap vanishes."""
            braket = np.vdot(a, b)
            norm = abs(braket)
            return braket / norm if norm > 1e-14 else 1.0

        def links(xs, ys):
            """x-links: (ix,iy)->(ix+1,iy); y-links: (ix,iy)->(ix,iy+1)."""
            nx, ny = len(xs), len(ys)
            ph_x = np.empty((nx, ny), dtype=complex)
            ph_y = np.empty((nx, ny), dtype=complex)
            for ix in range(nx):
                for iy in range(ny):
                    a = cache[(xs[ix], ys[iy])]
                    ph_x[ix, iy] = link(a, cache[(xs[(ix + 1) % nx], ys[iy])])
                    ph_y[ix, iy] = link(a, cache[(xs[ix], ys[(iy + 1) % ny])])
            return ph_x, ph_y

        for n_pass in range(self.flux_max_refine + 1):
            # one batch per pass: every grid point not solved yet, all independent of each other
            missing = [(x, y) for x in xs for y in ys if (x, y) not in cache]
            if missing:
                keys = list(cache)
                pts = np.array([(float(a), float(b)) for a, b in keys]).reshape(-1, 2)
                requests = []
                for x, y in missing:
                    v0 = None
                    if keys:  # warm start from the closest solved twist on the torus
                        d = np.abs(pts - (float(x), float(y)))
                        d = np.minimum(d, 1 - d)
                        v0 = cache[keys[int(np.argmin((d ** 2).sum(axis=1)))]]
                    requests.append((angle(x), angle(y), v0, False))
                results = yield requests
                for key, (psi, _) in zip(missing, results):
                    cache[key] = psi

            ph_x, ph_y = links(xs, ys)
            # curvature[ix,iy] = Berry flux through the plaquette whose lower-left corner is the
            # phi point (ix,iy): a=(ix,iy) b=(ix+1,iy) c=(ix+1,iy+1) d=(ix,iy+1)
            plaq = (ph_x * np.roll(ph_y, -1, axis=0)
                    * np.conj(np.roll(ph_x, -1, axis=1)) * np.conj(ph_y))
            curv = np.angle(plaq)
            hot = np.argwhere(np.abs(curv) > self.flux_max_angle)
            if len(hot) == 0 or n_pass == self.flux_max_refine:
                break
            xs = sorted(set(xs) | {midpoint(xs, int(ix)) for ix, _ in hot})
            ys = sorted(set(ys) | {midpoint(ys, int(iy)) for _, iy in hot})

        chern = curv.sum() / two_pi

        gap_min, gap_phi = np.nan, np.full(2, np.nan)
        if gap:
            # centre of the plaquette with the largest |curvature|
            ix, iy = np.unravel_index(np.argmax(np.abs(curv)), curv.shape)
            gap_phi = np.array([angle(midpoint(xs, int(ix))), angle(midpoint(ys, int(iy)))])
            # warm start from the plaquette corner
            results = yield [(gap_phi[0], gap_phi[1], cache[(xs[ix], ys[iy])], True)]
            gap_min = results[0][1]

        # Chern number and curvature on the grid of twist angles, for saving and plotting
        return dict(chern=chern, curvature=curv,
                    phi_x=np.array([angle(u) for u in xs]), phi_y=np.array([angle(u) for u in ys]),
                    gap=gap_min, gap_phi=gap_phi)

    def chern_number(self, delta, U, V, site_dis=None, bond_dis=None, gap=True):
        """Serial driver of chern_job: every solve runs on this process. Returns the chern_job dict."""
        job = self.chern_job(delta, U, V, site_dis, bond_dis, gap)
        try:
            requests = next(job)
            while True:
                results = [self.solve_twist(delta, U, V, *req, site_dis, bond_dis) for req in requests]
                requests = job.send(results)
        except StopIteration as stop:
            return stop.value
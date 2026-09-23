import numpy as np
from .lattice import NUM_SITES, SUB_SIGN
from .basis import BASIS
from .hamiltonian import hop_element, occupation_table
from .config import N_FLUX


class Observables:
    def __init__(self, solver):
        self.solver = solver
        self.occ_up = occupation_table(BASIS.up.states)
        self.occ_dn = self.occ_up if BASIS.dn is BASIS.up else occupation_table(BASIS.dn.states)
        self.stag_up = SUB_SIGN @ self.occ_up
        self.stag_dn = SUB_SIGN @ self.occ_dn

    def prob(self, psi):
        """|psi|^2 reshaped to (dim_dn, dim_up), normalized."""
        prob = np.abs(psi.reshape(BASIS.dim_dn, BASIS.dim_up)) ** 2
        return prob / prob.sum()

    def cdw(self, psi):
        """Staggered charge structure factor S_CDW = (1/N) sum_ij xi_i xi_j <n_i n_j>."""
        prob = self.prob(psi)
        stag_charge = self.stag_up[None, :] + self.stag_dn[:, None]
        return float(np.sum(stag_charge * stag_charge * prob) / NUM_SITES)

    def sdw(self, psi):
        """Staggered spin structure factor, same idea with n_up - n_dn."""
        prob = self.prob(psi)
        stag_spin = self.stag_up[None, :] - self.stag_dn[:, None]
        return float(np.sum(stag_spin * stag_spin * prob) / NUM_SITES)

    def density_matrix(self, psi):
        """
        rho_ij = <c_i^dagger c_j> per spin.
        """
        M = psi.reshape(BASIS.dim_dn, BASIS.dim_up)
        G = M.conj().T @ M     # (dim_up, dim_up)
        H2 = M @ M.conj().T    # (dim_dn, dim_dn)

        rho_up = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)
        rho_dn = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)

        diagG, diagH2 = np.diag(G).real, np.diag(H2).real
        for site in range(NUM_SITES):
            rho_up[site, site] = self.occ_up[site] @ diagG
            rho_dn[site, site] = self.occ_dn[site] @ diagH2

        for i in range(NUM_SITES):
            for j in range(NUM_SITES):
                if i == j:
                    continue
                for bra, ket, sign in hop_element(BASIS.up.states, j, i):
                    rho_up[j, i] += sign * G[ket, bra]
                for bra, ket, sign in hop_element(BASIS.dn.states, j, i):
                    rho_dn[j, i] += sign * H2[bra, ket]

        return rho_up, rho_dn

    r'''
    def resta_marker(self, psi):
        """
        Computes the Local Bott Index, which serves as the mathematically 
        correct Local Chern Marker (LCM) for Periodic Boundary Conditions.
        """
        import scipy.linalg

        rho_up, rho_dn = self.density_matrix(psi)
        P = rho_up + rho_dn
        
        # 1. Define the supercell area and reciprocal lattice vectors G1, G2
        # Supercell vectors are L1 = (6.0, 0.0), L2 = (1.5, 1.5*sqrt(3))
        area = 6.0 * 1.5 * np.sqrt(3)
        G1 = 2 * np.pi * np.array([1.5 * np.sqrt(3), -1.5]) / area
        G2 = 2 * np.pi * np.array([0.0, 6.0]) / area

        # 2. Map real-space coordinates to periodic phase angles
        theta1 = POSITIONS @ G1
        theta2 = POSITIONS @ G2
        
        # 3. Create diagonal unitary phase operators
        U1 = np.diag(np.exp(1j * theta1))
        U2 = np.diag(np.exp(1j * theta2))
        
        # 4. Project the unitaries into the occupied subspace (Q = I - P)
        Q = np.eye(NUM_SITES, dtype=complex) - P
        V1 = P @ U1 @ P + Q
        V2 = P @ U2 @ P + Q

        # 5. Form the closed-loop plaquette matrix
        M = V1 @ V2 @ V1.conj().T @ V2.conj().T
        
        # 6. Extract the local markers via the principal matrix logarithm
        # M is close to identity, so the log captures the geometric phase
        logM = scipy.linalg.logm(M)
        
        # The imaginary part of the diagonal elements gives the Local Bott Index
        local_markers = np.imag(np.diag(logM)) / (2 * np.pi)
        
        # For the Bott Index formulation, the TRACE (sum) is the global invariant,
        # yielding ~ 1.0 or -1.0 in the topological phase.
        return np.sum(local_markers)
    '''

    def chern_number(self, delta, U, V, grid=N_FLUX, v0=None, site_dis=None, bond_dis=None):
        """
        Total (up+down) Chern number via Fukui-Hatsugai-Suzuki flux
        integration, with adaptive refinement and corner caching.
        """
        self.flux_cache = {}
        angles = np.linspace(0, 2 * np.pi, grid, endpoint=False)
        curvature = 0.0
        for ix in range(grid):
            ix2 = (ix + 1) % grid
            for iy in range(grid):
                iy2 = (iy + 1) % grid
                curvature += self.adaptive_curvature(delta, U, V, angles[ix], angles[iy], angles[ix2], angles[iy2],
                                                     site_dis, bond_dis, v0, threshold=np.pi / 4, depth=0, max_depth=6)
        return curvature / (2 * np.pi)

    def gs_with_fluxcache(self, delta, U, V, fx, fy, site_dis, bond_dis, v0=None):
        key = (round(fx % (2 * np.pi), 9), round(fy % (2 * np.pi), 9))
        psi = self.flux_cache.get(key)
        if psi is None:
            _, psi = self.solver.ground_state(delta, U, V, fx, fy, v0, False, site_dis, bond_dis)
            self.flux_cache[key] = psi
        return psi

    def curvature(self, delta, U, V, fx0, fy0, fx1, fy1, site_dis, bond_dis, v0=None):
        corners = [(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]
        v = v0
        psi = []
        for fx, fy in corners:
            v = self.gs_with_fluxcache(delta, U, V, fx, fy, site_dis, bond_dis, v)
            psi.append(v)

        def phase(a, b):
            phi = np.vdot(a, b)
            return phi / abs(phi)

        curv = np.angle(phase(psi[0], psi[1]) * phase(psi[1], psi[2])
                        * np.conj(phase(psi[3], psi[2])) * np.conj(phase(psi[0], psi[3])))
        return curv, v

    def adaptive_curvature(self, delta, U, V, fx0, fy0, fx1, fy1, site_dis, bond_dis,
                            v0=None, threshold=np.pi / 4, depth=0, max_depth=2):
        curv, v0 = self.curvature(delta, U, V, fx0, fy0, fx1, fy1, site_dis, bond_dis, v0)
        if abs(curv) < threshold or depth >= max_depth:
            fxm, fym = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
            print((fxm, fym), curv)
            return curv
        fxm, fym = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
        total = 0.0
        for (a, b, c, d) in [(fx0, fy0, fxm, fym), (fxm, fy0, fx1, fym),
                            (fx0, fym, fxm, fy1), (fxm, fym, fx1, fy1)]:
            total += self.adaptive_curvature(delta, U, V, a, b, c, d, site_dis, bond_dis,
                                             v0=v0, threshold=threshold, depth=depth + 1, max_depth=max_depth)
        return total
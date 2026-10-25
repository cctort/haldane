import time
import psutil

from ed.config import Config
from ed.lattice import Lattice
from ed.basis import SpinfulBasis
from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

p = psutil.Process()
def info(t):
    print(f"time: {time.time()-t:.3f}s | memory: {p.memory_info().rss/1024**2:.1f} MB")

DELTA = 0.0
U = 8.
V = 0.0

cfg = Config(n_max=[2, 3], n_flux=16)

t = time.time()
lattice = Lattice(cfg)
basis = SpinfulBasis(cfg, lattice)
print(f"Lattice and basis built, N_SITES={lattice.n_sites}, dim={basis.dim},", end=" ")
info(t)

t = time.time()
hamiltonian = Hamiltonian(cfg, basis, lattice)
hamiltonian.build(DELTA, U, V)
print(f"Hamiltonian built, delta={DELTA:.5g}, U={U:.5g}, V={V:.5g},", end=" ")
info(t)

solver = EDSolver(cfg, hamiltonian)

t = time.time()
Egs, psi, gap = solver.ground_state(DELTA, U, V, gap=True)
print(f"Ground state found, Egs={Egs}, gap={gap}", end=" ")
info(t)

observables = Observables(cfg, basis, lattice, solver)

for name, corr_fn in (("corr_ch", observables.corr_ch), ("corr_sz", observables.corr_sz), ("corr_sx", observables.corr_sx)):
    t = time.time()
    corr = corr_fn(psi)
    print(f"{name}: S(q=0)={observables.S_q0(corr):.5g}, S(q=pi)={observables.S_qpi(corr):.5g},", end=" ")
    info(t)

t = time.time()
chern, max_angle = observables.chern_number(DELTA, U, V)
print(f"Chern number: {chern:.5g}, max angle: {max_angle:.5g}", end=" ")
info(t)
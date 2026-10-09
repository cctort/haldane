import time
import psutil

from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import *

p = psutil.Process()
def info(t):
    print(f"time: {time.time()-t:.3f}s | memory: {p.memory_info().rss/1024**2:.1f} MB")

DELTA = 0.0
U = 8.5
V = 0.0

t = time.time()
hamiltonian = Hamiltonian()
hamiltonian.build(DELTA, U, V)
print(f"Hamiltonian built, delta={DELTA:.5g}, U={U:.5g}, V={V:.5g},", end=" ")
info(t)

solver = EDSolver(hamiltonian)

t = time.time()
Egs, psi, gap = solver.ground_state(DELTA, U, V, gap=True)
print(f"Ground state found, Egs={Egs}, gap={gap}", end=" ")
info(t)

observables = Observables(solver)

t = time.time()
cdw = observables.cdw(psi)
print(f"CDW: {cdw:.5g},", end=" ")
info(t)

t = time.time()
sdw = observables.sdw(psi)
print(f"SDW: {sdw:.5g},", end=" ")
info(t)

N_FLUX = 16

t = time.time()
chern, curv = observables.chern_number(DELTA, U, V, N_FLUX)
print(f"Chern number: {chern:.5g},", end=" ")
info(t)

print('x = ', list(np.linspace(0, 2 * np.pi, N_FLUX, endpoint=False)))
print('y = ', list(np.linspace(0, 2 * np.pi, N_FLUX, endpoint=False)))
print('curvs = ', curv)

#t = time.time()
#rs = observables.resta_marker(psi)
#print(f"Resta marker: {rs},", end=" ")
#info(t)
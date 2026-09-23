import time
import psutil

from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import *

p = psutil.Process()
def info(t):
    print(f"time: {time.time()-t:.3f}s | memory: {p.memory_info().rss/1024**2:.1f} MB")

DELTA = 0.0
U = 8.0
V = 0.0

t = time.time()
hamiltonian = Hamiltonian()
hamiltonian.build(DELTA, U, V)
print(f"Hamiltonian built, delta={DELTA:.5g}, U={U:.5g}, V={V:.5g},", end=" ")
info(t)

solver = EDSolver(hamiltonian)

t = time.time()
Egs, psi = solver.ground_state(DELTA, U, V, gap=False)
print(f"Ground state found, Egs={Egs}, gap={0}", end=" ")
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

t = time.time()
chern = observables.chern_number(DELTA, U, V, 4)
print(f"Chern number: {chern:.5g},", end=" ")
info(t)

#t = time.time()
#rs = observables.resta_marker(psi)
#print(f"Resta marker: {rs},", end=" ")
#info(t)
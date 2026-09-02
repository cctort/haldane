import time
import psutil

from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import *

p = psutil.Process()
def info(t):
    print(f"time: {time.time()-t:.3f}s | memory: {p.memory_info().rss/1024**2:.1f} MB")

DELTA = 0.0
U = 5.0
V = 0.0

t = time.time()
hamiltonian = Hamiltonian()
print(f"Hamiltonian built, delta={DELTA:.5g}, U={U:.5g}, V={V:.5g},", end=" ")
info(t)

solver = EDSolver(hamiltonian)

t = time.time()
Egs, psi, gap = solver.ground_state(DELTA, U, V, gap=True)
print(f"Ground state found, Egs={Egs}, gap={gap}", end=" ")
#print(f"gap={gap:.5g}, rtol={rtol:.5g},", end=" ")
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

#t = time.time()
#chern = observables.chern_number(DELTA, U, V)
#print(f"Chern number: {chern:.5g},", end=" ")
#info(t)
import time
from collections import deque

import psutil
import numpy as np
from mpi4py import MPI

from ed.config import Config
from ed.lattice import Lattice
from ed.basis import SpinfulBasis
from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

p = psutil.Process()
def info(t):
    print(f"time: {time.time()-t:.3f}s | memory: {p.memory_info().rss/1024**2:.1f} MB", flush=True)

DELTA = 0.0
U = 7.75
V = 0.0

cfg = Config(n_max=[2, 3], n_flux=4)

# Every rank builds what it needs to solve one twist; only rank 0 prints and runs the serial steps.
t = time.time()
lattice = Lattice(cfg)
basis = SpinfulBasis(cfg, lattice)
if rank == 0:
    print(f"Lattice and basis built, N_SITES={lattice.n_sites}, dim={basis.dim}, MPI ranks={size},", end=" ")
    info(t)

t = time.time()
hamiltonian = Hamiltonian(cfg, basis, lattice)
if rank == 0:
    hamiltonian.build(DELTA, U, V)   # only to time it, the solver builds what it needs itself
    print(f"Hamiltonian built, delta={DELTA:.5g}, U={U:.5g}, V={V:.5g},", end=" ")
    info(t)

solver = EDSolver(cfg, hamiltonian)
observables = Observables(cfg, basis, lattice, solver)

def serve_twists():
    """Workers: solve one twist per message from rank 0 until told to stop."""
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 99:
            return
        idx, phi_x, phi_y, v0, want_gap = task
        out = observables.solve_twist(DELTA, U, V, phi_x, phi_y, v0, want_gap)
        comm.send((idx, out), dest=0, tag=0)


def chern_parallel(gap=True):
    """Rank 0: run the adaptive Chern calculation, farming every batch of independent twist solves
    out to the workers. A batch is one refinement pass (the initial grid, then the new points of each
    refinement), plus a final single solve for the gap. Returns (result dict, n_solves, n_batches)."""
    n_workers = size - 1
    job = observables.chern_job(DELTA, U, V, gap=gap)
    n_solves = n_batches = 0
    try:
        requests = next(job)
        while True:
            n_batches += 1
            n_solves += len(requests)
            results = [None] * len(requests)
            todo = deque(enumerate(requests))
            idle, in_flight = deque(range(1, n_workers + 1)), 0
            while todo or in_flight:
                while idle and todo:
                    idx, (phi_x, phi_y, v0, want_gap) = todo.popleft()
                    comm.send((idx, phi_x, phi_y, v0, want_gap), dest=idle.popleft(), tag=11)
                    in_flight += 1
                status = MPI.Status()
                idx, out = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                results[idx] = out
                idle.append(status.Get_source())
                in_flight -= 1
            requests = job.send(results)
    except StopIteration as stop:
        return stop.value, n_solves, n_batches


if rank != 0:
    serve_twists()
    raise SystemExit

try:
    t = time.time()
    Egs, psi, gap = solver.ground_state(DELTA, U, V, gap=True)
    print(f"Ground state found, Egs={Egs}, gap={gap}", end=" ")
    info(t)

    for name, corr_fn in (("corr_ch", observables.corr_ch), ("corr_sz", observables.corr_sz), ("corr_sx", observables.corr_sx)):
        t = time.time()
        corr = corr_fn(psi)
        print(f"{name}: S(q=0)={observables.S_q0(corr):.5g}, S(q=pi)={observables.S_qpi(corr):.5g},", end=" ")
        info(t)

    t = time.time()
    if size == 1:   # plain `python run_single.py`: everything on this process
        res = observables.chern_number(DELTA, U, V)
        stats = ""
    else:
        res, n_solves, n_batches = chern_parallel()
        stats = f"\n{n_solves} twist solves in {n_batches} sequential batches on {size - 1} workers,"
    gap_phi = res['gap_phi']
    nx, ny = len(res['phi_x']), len(res['phi_y'])
    print(f"Chern number: {res['chern']:.5g}, max |curvature|: {np.abs(res['curvature']).max():.5g}, "
          f"({nx}, {ny}) flux grid,{stats}\n"
          f"gap at max-curvature plaquette: {res['gap']:.5g} at ({gap_phi[0]:.3g}, {gap_phi[1]:.3g})", end=" ")
    info(t)

    np.savez(f"chern_grid_U{U:.5g}_ref.npz", **res)
finally:
    for w in range(1, size):   # release the workers, also if something above failed
        comm.send(None, dest=w, tag=99)
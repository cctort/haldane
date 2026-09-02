import time
import h5py
from pathlib import Path
from mpi4py import MPI
import numpy as np

from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

DATA_DIR = Path('./data')
RESULTS_FILE = 'phase_diagram.h5'

DELTA_RANGE = np.linspace(0, 4, 32)
U_RANGE = np.linspace(0, 13, 32)
V_RANGE = np.linspace(0, 4, 32)

CHERN = True
GAP = True

def chunk(points, rank, size):
    """Contiguous, near-equal split of `points` -- the piece for `rank`."""
    n = len(points)
    base, rem = divmod(n, size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return points[start:end]


def sweep(points, key_fn, solver, observables, rank):
    """
    Run ground_state + cdw/sdw/chern over `points` (already chunked to
    this rank), warm-starting each solve from the previous point.

    key_fn(point) -> (delta, U, V) used both to call the solver and as
    the results dict key.
    """
    results = {}
    v0 = None
    t0 = time.time()

    for point in points:
        delta, U, V = key_fn(point)
        res = solver.ground_state(delta, U, V, v0=v0, gap=GAP)
        if GAP:
            E, psi, gap = res
        else:
            E, psi = res
        v0 = psi.reshape(-1, 1)

        if CHERN:
            try:
                chern = observables.chern_number(delta, U, V)
            except RuntimeError:
                chern = None

        results[(delta, U, V)] = {
            'E_gs': E,
            'cdw': observables.cdw(psi),
            'sdw': observables.sdw(psi),
            'gap': gap if GAP else None,
            'chern': chern if CHERN else None,
        }

    print(f"rank {rank}: {len(points)} points in {time.time()-t0:.1f}s")
    return results


def save(results, path):
    with h5py.File(path, 'w') as f:
        for (delta, U, V), data in results.items():
            grp = f.create_group(f"delta_{delta:.3f}_U_{U:.3f}_V_{V:.3f}")
            for k, v in data.items():
                if v is not None:
                    grp.attrs[k] = v


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    hamiltonian = Hamiltonian()
    solver = EDSolver(hamiltonian)
    observables = Observables(solver)

    # Fig. 1(a): (delta, U) at V = 0
    points_a = [(d, u) for d in DELTA_RANGE for u in U_RANGE]
    my_points_a = chunk(points_a, rank, size)
    results_a = sweep(my_points_a, lambda p: (p[0], p[1], 0.0), solver, observables, rank)

    # Fig. 1(b): (U, V) at delta = 0
    points_b = [(u, v) for u in U_RANGE for v in V_RANGE]
    my_points_b = chunk(points_b, rank, size)
    results_b = sweep(my_points_b, lambda p: (0.0, p[0], p[1]), solver, observables, rank)

    all_a = comm.gather(results_a, root=0)
    all_b = comm.gather(results_b, root=0)

    if rank == 0:
        results = {}
        for r in all_a + all_b:
            results.update(r)

        DATA_DIR.mkdir(exist_ok=True)
        out_path = DATA_DIR / RESULTS_FILE
        save(results, out_path)
        print(f"\nSaved {len(results)} points to {out_path}")


if __name__ == "__main__":
    main()
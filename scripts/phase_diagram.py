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

SWEEP_VARS = ('delta', 'U')
FIXED_VARS = {'V': 0.0}

RANGES = {
    'delta': np.linspace(0, 4, 50),
    'U': np.linspace(0, 13, 50),
    'V': np.linspace(0, 4, 2)
}

CHERN = True
GAP = True

def chunk(points, rank, size):
    n = len(points)
    base, rem = divmod(n, size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return points[start:end]

def make_point(x, y):
    d = {**FIXED_VARS, SWEEP_VARS[0]: x, SWEEP_VARS[1]: y}
    return (d.get('delta', 0.0), d.get('U', 0.0), d.get('V', 0.0))

def sweep(points, solver, observables, rank):
    results = {}
    v0 = None
    t0 = time.time()

    for (x, y) in points:
        delta, U, V = make_point(x, y)
        res = solver.ground_state(delta, U, V, v0=v0, gap=GAP)
        
        E, psi = res[:2]
        gap = res[2] if GAP else None
        v0 = psi.reshape(-1, 1)

        chern = None
        if CHERN:
            try:
                chern = observables.chern_number(delta, U, V, grid=4)
            except RuntimeError:
                chern = np.nan

        results[(delta, U, V)] = {
            'E_gs': E,
            'cdw': observables.cdw(psi),
            'sdw': observables.sdw(psi),
            'gap': gap,
            'chern': chern,
        }

    print(f"rank {rank}: {len(points)} points in {time.time()-t0:.1f}s")
    return results

def save(results, path):
    with h5py.File(path, 'a') as f:
        for (delta, U, V), data in results.items():
            grp = f.require_group(f"delta_{delta:.4f}_U_{U:.4f}_V_{V:.4f}")
            grp.attrs.update({'delta': delta, 'U': U, 'V': V})
            for k, v in data.items():
                if v is not None:
                    grp.attrs[k] = v

def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    solver = EDSolver(Hamiltonian())
    observables = Observables(solver)

    all_points = [(x, y) for x in RANGES[SWEEP_VARS[0]] for y in RANGES[SWEEP_VARS[1]]]
    my_points = chunk(all_points, rank, size)
    
    my_results = sweep(my_points, solver, observables, rank)
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        results = {k: v for r in all_results for k, v in r.items()}
        DATA_DIR.mkdir(exist_ok=True)
        save(results, DATA_DIR / RESULTS_FILE)

if __name__ == "__main__":
    main()
import time
import json
import heapq
import itertools
import os
import hashlib
from collections import deque
from pathlib import Path
import h5py
from mpi4py import MPI
import numpy as np

from ed.config import Config
from ed.lattice import Lattice
from ed.basis import SpinfulBasis
from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

cfg = Config()

# Lattice
cfg.cell = 'honeycomb'
cfg.cluster = 'A'
cfg.n_max = [3, 2]

# Electron filling and total spin
cfg.filling = 0.5
cfg.Sz = 0.0

# Physical parameters
cfg.t1 = 1.0
cfg.t2 = 0.2
cfg.phi = 0.5

# ED tolerance and Chern number flux grid
cfg.tol = 0
cfg.n_flux = 2
cfg.flux_max_refine = 4
cfg.flux_max_angle = 0.3

# Disorder
cfg.n_samples = 1
cfg.w0 = 0.0
cfg.w1 = 0.0
cfg.w2 = 0.0
SEED = 123 

CLEAN = cfg.w0 == cfg.w1 == cfg.w2 == 0
if CLEAN:
    cfg.n_samples = 1  # all realizations would be identical

DATA_DIR = Path('./data/ed_diagrams')
RESULTS_FILE = 'sweep.h5'
REGISTRY_FILE = 'configs.json'

SWEEP_VARS = ('U', 'V')
FIXED_VARS = {'delta': 0.0}

RANGES = {'delta': np.linspace(0, 4, 14),
          'U': np.linspace(0, 12.5, 14),
          'V': np.linspace(0, 4, 14)}

# Possible observables: 'E_gs', 'gap', 'corr_ch', 'corr_sz', 'corr_sx', 'chern', 'min_gap'
# ('min_gap' = gap at the largest-curvature plaquette, computed together with 'chern')
OBS = ['chern']
REPLACE_OBS = False

# Parallel Chern scheduling: how many (point, sample) Chern calculations rank 0 keeps in flight.
# Each one holds its ground states in rank-0 memory (up to ~ (n_flux * 2**flux_max_refine)^2 vectors),
# so lower this if rank 0 runs out of memory; it must stay large enough to keep all workers fed.
MAX_ACTIVE_JOBS = None   # None -> one per worker

GS_OBS = ('E_gs', 'gap', 'corr_ch', 'corr_sz', 'corr_sx')
NEED_GS = any(o in OBS for o in GS_OBS)   # observables that need the ground state at the default twist
DO_CHERN = 'chern' in OBS
WANT_GAP = 'min_gap' in OBS


def disorder_config(lattice):
    if CLEAN:  # no disorder objects at all, the solver gets None
        return [None] * cfg.n_samples, [None] * cfg.n_samples

    site_dis_list = []
    bond_dis_list = []
    
    for _ in range(cfg.n_samples):
        site_dis_list.append((np.random.rand(lattice.n_sites) - 0.5) * cfg.w0)
        sample_bonds = {
            (i, j): (np.random.rand() - 0.5) * cfg.w1 
            for i, j, _ in lattice.bonds[1]
        }
        sample_bonds.update({
            (i, j): (np.random.rand() - 0.5) * cfg.w2 
            for i, j, _ in lattice.bonds[2]
        })
        bond_dis_list.append(sample_bonds)

    return site_dis_list, bond_dis_list


def fmt_ranks(ranks):
    """{1,2,3,5,7,8} -> '1-3,5,7-8'"""
    out, start, prev = [], None, None
    for r in sorted(ranks) + [None]:
        if start is None:
            start = prev = r
        elif r is not None and r == prev + 1:
            prev = r
        else:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = r
    return ",".join(out)


def log_done(label, pt, s, dt, grid=None, n_solves=None, ranks=None):
    delta, U, V = pt
    d_vals = {'delta': delta, 'U': U, 'V': V}
    sweep_str = ", ".join([f"{var}={d_vals[var]:.4f}" for var in SWEEP_VARS])
    extra = ""
    if n_solves is not None:
        extra += f", {n_solves} solves"
    if grid is not None:
        extra += f", ({grid[0]}, {grid[1]}) flux grid"
    if ranks:
        extra += f", ranks {fmt_ranks(ranks)}"
    print(f"{label}: ({sweep_str}) sample {s} done in {dt:.1f}s{extra}", flush=True)


def run_sample(pt, s, site_dis, bond_dis, solver, observables, rank, with_chern=True):
    """Everything for one (point, sample). In parallel runs the Chern number is split off
    (with_chern=False) and handled by the scheduler on rank 0, one twist solve at a time."""
    t0 = time.time()

    delta, U, V = pt

    data = {}
    if NEED_GS:
        res = solver.ground_state(delta, U, V, gap='gap' in OBS,
                                  site_dis=site_dis, bond_dis=bond_dis)
        E, psi = res[0], res[1]

        if 'E_gs' in OBS:
            data['E_gs'] = E
        if 'gap' in OBS:
            data['gap'] = res[2]
        if 'corr_ch' in OBS:
            data['corr_ch'] = observables.corr_ch(psi)
        if 'corr_sz' in OBS:
            data['corr_sz'] = observables.corr_sz(psi)
        if 'corr_sx' in OBS:
            data['corr_sx'] = observables.corr_sx(psi)

    grid = None
    if with_chern and DO_CHERN:
        chern_res = observables.chern_number(delta, U, V, site_dis=site_dis, bond_dis=bond_dis, gap=WANT_GAP)
        data['chern'] = chern_res['chern']
        if WANT_GAP:
            data['min_gap'] = chern_res['gap']
        grid = (len(chern_res['phi_x']), len(chern_res['phi_y']))

    log_done(f"rank {rank}", pt, s, time.time() - t0, grid)

    return data


def assemble(samples):
    """samples: {point: {sample index: data}} -> {point: {obs: list over realizations}}"""
    results = {}
    for pt, by_sample in samples.items():
        ordered = [by_sample[s] for s in range(cfg.n_samples)]
        results[pt] = {k: [d[k] for d in ordered] for k in OBS}
        if cfg.n_samples == 1:  # no list for a single realization
            results[pt] = {k: v[0] for k, v in results[pt].items()}
    return results


def get_current_config():
    # Maps uppercase keys (for metadata storage) to lowercase attributes in the config object
    return {'CELL': cfg.cell, 'CLUSTER': cfg.cluster, 'N_MAX': cfg.n_max,
            'FILLING': cfg.filling, 'SZ': cfg.Sz,
            'T1': cfg.t1, 'T2': cfg.t2, 'PHI': cfg.phi,
            'TOL': cfg.tol, 'N_FLUX': cfg.n_flux,
            'N_SAMPLES': cfg.n_samples, 'W0': cfg.w0, 'W1': cfg.w1, 'W2': cfg.w2}


def get_config_hash(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]


def update_registry(data_dir, h5_filename, current_config):
    registry_path = data_dir / REGISTRY_FILE
    registry = {}
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass
    registry[h5_filename] = current_config
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)


def get_or_create_hdf5_file(data_dir, base_filename, current_config):
    config_hash = get_config_hash(current_config)
    filename = f"{Path(base_filename).stem}_{config_hash}{Path(base_filename).suffix}"
    path = data_dir / filename

    if not path.exists():
        with h5py.File(path, 'w') as f:
            for k, v in current_config.items():
                f.attrs[k] = v
            
    update_registry(data_dir, filename, current_config)
    return path


def save(results, path):
    with h5py.File(path, 'a') as f:
        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                for key in OBS:
                    grp.attrs.pop(key, None)

        for (delta, U, V), data in results.items():
            grp_name = f"delta_{delta:.4f}_U_{U:.4f}_V_{V:.4f}"
            grp = f.require_group(grp_name)
            grp.attrs['delta'] = delta
            grp.attrs['U'] = U
            grp.attrs['V'] = V
            for k, v in data.items():
                grp.attrs[k] = v

        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                if len([k for k in grp.attrs.keys() if k not in ('delta', 'U', 'V')]) == 0:
                    del f[grp_name]


class ChernJob:
    """One adaptive-grid Chern calculation ((point, sample) task), driven by rank 0."""

    def __init__(self, gen, pt, s, site_dis, bond_dis):
        self.gen, self.pt, self.s = gen, pt, s
        self.site_dis, self.bond_dis = site_dis, bond_dis
        self.pending, self.results, self.n_solves = 0, [], 0
        self.ranks = set()   # workers that solved at least one twist of this job
        self.t0 = time.time()


def master(comm, tasks, observables):
    """Rank 0: dynamic scheduler over individual work units, workers never wait for a whole task.

    Work units:
      ('sample', ...)  non-Chern observables of one (point, sample)       -> run_sample on a worker
      ('twist', ...)   ground state at one flux point of one Chern job    -> solve_twist on a worker
    A Chern job (Observables.chern_job) yields one batch of independent twist solves per refinement
    pass. Rank 0 collects the batch, evaluates links / curvature / refinement (cheap), and queues the
    next batch, so a point that needs heavy refinement spreads over all free workers. Work of the
    oldest tasks is dispatched first, which keeps the number of unfinished jobs (memory) small and
    shortens the tail at the end of the sweep.
    """
    n_workers = comm.Get_size() - 1
    max_active = MAX_ACTIVE_JOBS or n_workers

    samples = {}
    ready, tiebreak = [], itertools.count()   # heap of (task id, tiebreak, work item)
    jobs = {}                                 # task id -> ChernJob
    todo = iter(enumerate(tasks))

    def push(tid, item):
        heapq.heappush(ready, (tid, next(tiebreak), item))

    def submit(tid, requests):
        job = jobs[tid]
        job.pending, job.results = len(requests), [None] * len(requests)
        job.n_solves += len(requests)
        for idx, (phi_x, phi_y, v0, want_gap) in enumerate(requests):
            push(tid, ('twist', tid, idx, job.pt, job.site_dis, job.bond_dis, phi_x, phi_y, v0, want_gap))

    def start_tasks():
        while len(jobs) < max_active:
            nxt = next(todo, None)
            if nxt is None:
                return
            tid, (pt, s, site_dis, bond_dis) = nxt
            if NEED_GS:
                push(tid, ('sample', pt, s, site_dis, bond_dis))
            if DO_CHERN:
                gen = observables.chern_job(*pt, site_dis=site_dis, bond_dis=bond_dis, gap=WANT_GAP)
                jobs[tid] = ChernJob(gen, pt, s, site_dis, bond_dis)
                submit(tid, next(gen))

    idle = deque(range(1, n_workers + 1))
    sent_at, busy = {}, 0.0
    t_start = time.time()
    start_tasks()

    while True:
        while idle and ready:
            _, _, item = heapq.heappop(ready)
            w = idle.popleft()
            comm.send(item, dest=w, tag=11)
            sent_at[w] = time.time()
        if len(idle) == n_workers:   # nothing queued, nothing in flight
            break

        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        w = status.Get_source()
        busy += time.time() - sent_at[w]
        idle.append(w)

        if msg[0] == 'sample':
            _, pt, s, data = msg
            samples.setdefault(pt, {}).setdefault(s, {}).update(data)
            continue

        _, tid, idx, out = msg
        job = jobs[tid]
        job.ranks.add(w)
        job.results[idx] = out
        job.pending -= 1
        if job.pending:
            continue
        try:
            submit(tid, job.gen.send(job.results))   # next refinement batch
        except StopIteration as stop:
            res = stop.value
            data = {'chern': res['chern']}
            if WANT_GAP:
                data['min_gap'] = res['gap']
            samples.setdefault(job.pt, {}).setdefault(job.s, {}).update(data)
            log_done("chern job", job.pt, job.s, time.time() - job.t0,
                     (len(res['phi_x']), len(res['phi_y'])), job.n_solves, job.ranks)
            del jobs[tid]
            start_tasks()

    for w in range(1, n_workers + 1):
        comm.send(None, dest=w, tag=99)
    utilization = busy / (n_workers * (time.time() - t_start))
    return samples, utilization


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    total_t0 = time.time()

    current_config = get_current_config()
    config_hash = get_config_hash(current_config)
    
    if rank == 0:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
        print("="*40)
        print(f"SLURM_JOB_ID : {slurm_job_id}")
        print(f"CONFIG HASH : {config_hash}")
        print(f"TARGET FILE : {RESULTS_FILE.replace('.h5', f'_{config_hash}.h5')}")
        print(f"CELL: {cfg.cell}, CLUSTER: {cfg.cluster}, N_MAX: {cfg.n_max},")
        print(f"FILLING: {cfg.filling:.3g}, SZ: {cfg.Sz:.3g},")
        print(f"T1: {cfg.t1:.3g}, T2: {cfg.t2:.3g}, PHI: {cfg.phi:.3g},")
        print(f"TOL: {cfg.tol}, N_FLUX: {cfg.n_flux},")
        print(f"N_SAMPLES: {cfg.n_samples}, W0: {cfg.w0:.3g}, W1: {cfg.w1:.3g}, W2: {cfg.w2:.3g}")
        print("FIXED PARAMS : ", FIXED_VARS)
        print("VARYING PARAMS : ", {var : (float(RANGES[var][0]), float(RANGES[var][-1]), len(RANGES[var])) for var in SWEEP_VARS})
        print("EVALUATING OBSERVABLES : ", OBS)
        DATA_DIR.mkdir(exist_ok=True)

    lattice = Lattice(cfg)
    if rank == 0:
        print(f'N_SITES : {lattice.n_sites}')
        print("="*40, flush=True)

    basis = SpinfulBasis(cfg, lattice)
    hamiltonian = Hamiltonian(cfg, basis, lattice)
    solver = EDSolver(cfg, hamiltonian)
    observables = Observables(cfg, basis, lattice, solver)

    comm.Barrier()

    inner_var = SWEEP_VARS[0]
    outer_vars = list(SWEEP_VARS[1:])
    
    outer_ranges = [RANGES[var] for var in outer_vars] if outer_vars else [[0]]
    inner_range = RANGES[inner_var]

    # Split to exactly 1 point per chunk
    inner_chunks = np.array_split(inner_range, len(inner_range))

    trajectories = []
    for outer_vals in itertools.product(*outer_ranges):
        for chunk in inner_chunks:
            if len(chunk) == 0:
                continue
            
            traj = []
            for inner_val in chunk:
                sweep_vals = {inner_var: inner_val}
                for v_name, v_val in zip(outer_vars, outer_vals):
                    sweep_vals[v_name] = v_val
                
                # We assemble the exact 3-element tuple right here
                d = {**FIXED_VARS, **sweep_vals}
                traj.append((d.get('delta', 0.0), d.get('U', 0.0), d.get('V', 0.0)))
                
            trajectories.append(traj)

    # one task per (point, disorder realization), with fresh disorder drawn for every point
    np.random.seed(SEED)
    tasks = []
    for traj in trajectories:
        for pt in traj:
            site_dis_list, bond_dis_list = disorder_config(lattice)
            for s in range(cfg.n_samples):
                tasks.append((pt, s, site_dis_list[s], bond_dis_list[s]))

    if size == 1:
        samples = {}
        for pt, s, site_dis, bond_dis in tasks:
            samples.setdefault(pt, {})[s] = run_sample(pt, s, site_dis, bond_dis, solver, observables, rank=0)
        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(assemble(samples), filepath)
        print(f"Successfully saved data to {filepath}")
        print(f"Total time elapsed: {time.time() - total_t0:.1f}s", flush=True)
        return

    if rank == 0:
        samples, utilization = master(comm, tasks, observables)
        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(assemble(samples), filepath)
        print(f"Successfully saved data to {filepath}")
        print(f"Worker utilization: {utilization:.1%} (time workers spent busy, message transfer included)")
        print(f"Total time elapsed: {time.time() - total_t0:.1f}s", flush=True)

    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            
            if status.Get_tag() == 99:
                break

            if task[0] == 'sample':
                _, pt, s, site_dis, bond_dis = task
                data = run_sample(pt, s, site_dis, bond_dis, solver, observables, rank, with_chern=False)
                comm.send(('sample', pt, s, data), dest=0, tag=0)
            else:
                _, tid, idx, pt, site_dis, bond_dis, phi_x, phi_y, v0, want_gap = task
                out = observables.solve_twist(*pt, phi_x, phi_y, v0, want_gap, site_dis, bond_dis)
                comm.send(('twist', tid, idx, out), dest=0, tag=0)


if __name__ == "__main__":
    main()
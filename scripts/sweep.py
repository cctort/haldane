import time
import json
import itertools
import os
import hashlib
from pathlib import Path
import h5py
from mpi4py import MPI
import numpy as np

from ed import config
from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

DATA_DIR = Path('./data/tuning')
RESULTS_FILE = 'sweep.h5'
REGISTRY_FILE = 'configs.json'

# Override the default parameters in config.py
CONFIG = {
    'NUM_SITES': 12,
    'NUM_ELECTRONS': 12,
    'N_UP': 6,
    'N_DN': 6,
    'T1': 1.0,
    'T2': 0.2,
    'PHI': 0.5,
    'TOL': 1e-7,
    'N_FLUX': 4,
}
for key, value in CONFIG.items():
    setattr(config, key, value)

SWEEP_VARS = ('U',)
FIXED_VARS = {'V': 0.0, 'delta': 0.0}

RANGES = {
    'delta': np.linspace(0, 4, 20),
    'U': np.linspace(6.75, 8.5, 64),
    'V': np.linspace(0, 4, 2)
}

# Options: 'E_gs', 'cdw', 'sdw', 'gap', 'chern'
OBS = ['E_gs', 'cdw', 'sdw', 'gap', 'chern']

# if True removes all previous values for the current OBS elements
REPLACE_OBS = False


def chunk(points, rank, size):
    n = len(points)
    base, rem = divmod(n, size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return points[start:end]


def make_point(sweep_values):
    sweep_dict = dict(zip(SWEEP_VARS, sweep_values))
    d = {**FIXED_VARS, **sweep_dict}
    return (d.get('delta', 0.0), d.get('U', 0.0), d.get('V', 0.0))


def sweep(points, solver, observables, rank):
    results = {}
    v0 = None
    t0 = time.time()
    
    calc_gap = 'gap' in OBS

    for pt in points:
        delta, U, V = make_point(pt)
        res = solver.ground_state(delta, U, V, v0=v0, gap=calc_gap)

        E, psi = res[:2]
        v0 = psi.reshape(-1, 1)
        
        pt_data = {}
        
        if 'E_gs' in OBS:
            pt_data['E_gs'] = E
            
        if 'gap' in OBS:
            pt_data['gap'] = res[2]
            
        if 'cdw' in OBS:
            pt_data['cdw'] = observables.cdw(psi)
            
        if 'sdw' in OBS:
            pt_data['sdw'] = observables.sdw(psi)
            
        if 'chern' in OBS:
            pt_data['chern'] = observables.chern_number(delta, U, V, config.N_FLUX)

        results[(delta, U, V)] = pt_data

    print(f"rank {rank}: {len(points)} points in {time.time()-t0:.1f}s")
    return results


def get_current_config():
    return {
        'NUM_SITES': config.NUM_SITES,
        'NUM_ELECTRONS': config.NUM_ELECTRONS,
        'N_UP': config.N_UP,
        'N_DN': config.N_DN,
        'T1': config.T1,
        'T2': config.T2,
        'PHI': config.PHI,
        'TOL': config.TOL,
        'N_FLUX': config.N_FLUX,
    }


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
    stem = Path(base_filename).stem
    ext = Path(base_filename).suffix
    
    filename = f"{stem}_{config_hash}{ext}"
    path = data_dir / filename

    if not path.exists():
        with h5py.File(path, 'w') as f:
            for k, v in current_config.items():
                f.attrs[k] = v
            
    update_registry(data_dir, filename, current_config)
    return path


def save(results, path):
    with h5py.File(path, 'a') as f:
        
        # Clear the current run's observables from ALL existing points
        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                for key in OBS:
                    if key in grp.attrs:
                        del grp.attrs[key]

        # Save the new results
        for (delta, U, V), data in results.items():
            grp_name = f"delta_{delta:.4f}_U_{U:.4f}_V_{V:.4f}"
            grp = f.require_group(grp_name)
            
            grp.attrs['delta'] = delta
            grp.attrs['U'] = U
            grp.attrs['V'] = V
            
            for k, v in data.items():
                if REPLACE_OBS:
                    grp.attrs[k] = v
                else:
                    #Only overwrite if the attribute does not already exist
                    if k not in grp.attrs:
                        grp.attrs[k] = v
                        
        # Clean up ghost groups with only 'delta', 'U', 'V' stored
        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                if len(grp.keys()) == 0 and all(k in ['delta', 'U', 'V'] for k in grp.attrs.keys()):
                    del f[grp_name]


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    current_config = get_current_config()
    config_hash = get_config_hash(current_config)
    
    if rank == 0:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
        print("="*40)
        print(f"SLURM_JOB_ID  : {slurm_job_id}")
        print(f"CONFIG HASH   : {config_hash}")
        print(f"TARGET FILE   : {RESULTS_FILE.replace('.h5', f'_{config_hash}.h5')}")
        print("CONFIGURATION :")
        print(json.dumps(current_config, indent=2))
        print("="*40, flush=True)

        DATA_DIR.mkdir(exist_ok=True)

    comm.Barrier()

    solver = EDSolver(Hamiltonian())
    observables = Observables(solver)

    sweep_arrays = [RANGES[var] for var in SWEEP_VARS]
    all_points = list(itertools.product(*sweep_arrays))
    
    my_points = chunk(all_points, rank, size)

    my_results = sweep(my_points, solver, observables, rank)
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        results = {k: v for r in all_results for k, v in r.items()}
        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(results, filepath)
        print(f"Successfully saved data to {filepath}")


if __name__ == "__main__":
    main()
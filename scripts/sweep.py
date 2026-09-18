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
from ed.lattice import Lattice
from ed.hamiltonian import Hamiltonian
from ed.ed_solver import EDSolver
from ed.observables import Observables

DATA_DIR = Path('./data/clean')
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
    'TOL': 0,
    'N_FLUX': 10,
    'N_SAMPLES': 1,
    'W0': 0.0,
    'W1': 0.0,
    'W2': 0.0
}
for key, value in CONFIG.items():
    setattr(config, key, value)

SWEEP_VARS = ('U', 'V')
FIXED_VARS = {'delta': 0.0}

RANGES = {
    'delta': np.linspace(0, 4, 21),
    'U': np.linspace(0, 12.5, 21),
    'V': np.linspace(0, 4, 21)
}

# Options: 'E_gs', 'cdw', 'sdw', 'gap', 'chern'
OBS = ['chern']

# if True removes all previous values for the current OBS elements
REPLACE_OBS = True


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


def sweep(points, lattice, solver, observables, rank):
    results = {}
    t0 = time.time()
    calc_gap = 'gap' in OBS

    for pt in points:
        delta, U, V = make_point(pt)
        
        # Generate disorder configurations for this parameter point
        site_dis_list, bond_dis_list = disorder_config(lattice)
        
        # Lists to store metrics across realizations
        sample_data = {obs_key: [] for obs_key in OBS}

        for s in range(config.N_SAMPLES):
            # Pass disorder configurations directly into ground_state
            res = solver.ground_state(
                delta, U, V, 
                gap=calc_gap, 
                v0=None,
                site_dis=site_dis_list[s],
                bond_dis=bond_dis_list[s]
            )

            E, psi = res[0], res[1]
            
            if 'E_gs' in OBS:
                sample_data['E_gs'].append(E)
            if 'gap' in OBS:
                sample_data['gap'].append(res[2])
            if 'cdw' in OBS:
                sample_data['cdw'].append(observables.cdw(psi))
            if 'sdw' in OBS:
                sample_data['sdw'].append(observables.sdw(psi))
            if 'resta' in OBS:
                sample_data['resta'].append(observables.resta_marker(psi))
            if 'chern' in OBS:
                sample_data['chern'].append(observables.chern_number(delta, U, V, grid=config.N_FLUX, 
                                             site_dis=site_dis_list[s], bond_dis=bond_dis_list[s]))

        # Store mean and standard deviation for each observable
        pt_data = {}
        for key, vals in sample_data.items():
            pt_data[key] = np.mean(vals)
            pt_data[f"{key}_std"] = np.std(vals)

        results[(delta, U, V)] = pt_data

    print(f"rank {rank}: {len(points)} points in {time.time()-t0:.1f}s")
    return results


def disorder_config(lattice):
    site_dis_list = []
    bond_dis_list = []
    
    for _ in range(config.N_SAMPLES):
        # Site disorder in [-W0/2, W0/2]
        site_dis_list.append((np.random.rand(config.NUM_SITES) - 0.5) * config.W0)
        
        # Bond disorder: NN bonds (W1) and NNN bonds (W2)
        sample_bonds = {
            (i, j): (np.random.rand() - 0.5) * config.W1 
            for i, j, _ in lattice.nn_bonds
        }
        sample_bonds.update({
            (i, j): (np.random.rand() - 0.5) * config.W2 
            for i, j, _ in lattice.nnn_bonds
        })
        bond_dis_list.append(sample_bonds)

    return site_dis_list, bond_dis_list


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
        'N_SAMPLES': config.N_SAMPLES,
        'W0': config.W0,
        'W1': config.W1,
        'W2': config.W2
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
        
        # When REPLACE_OBS is True, clear active OBS from ALL points across the entire file first
        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                for key in OBS:
                    if key in grp.attrs:
                        del grp.attrs[key]

        # Overwrite specified observables for the points in the current run
        for (delta, U, V), data in results.items():
            grp_name = f"delta_{delta:.4f}_U_{U:.4f}_V_{V:.4f}"
            grp = f.require_group(grp_name)
            
            grp.attrs['delta'] = delta
            grp.attrs['U'] = U
            grp.attrs['V'] = V
            
            # Unconditionally write/overwrite active observables for evaluated points
            for k, v in data.items():
                grp.attrs[k] = v

        # When REPLACE_OBS is True, remove points that no longer hold any observables
        if REPLACE_OBS:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                obs_attrs = [k for k in grp.attrs.keys() if k not in ('delta', 'U', 'V')]
                if len(obs_attrs) == 0:
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
        print("FIXED PARAMS : ", FIXED_VARS)
        print("VARYING PARAMS : ", {var : (float(RANGES[var][0]), float(RANGES[var][-1]), len(RANGES[var])) for var in SWEEP_VARS})
        print("EVALUATING OBSERVABLES : ", OBS)
        print("="*40, flush=True)

        DATA_DIR.mkdir(exist_ok=True)

    comm.Barrier()

    lattice = Lattice()
    solver = EDSolver(Hamiltonian())
    observables = Observables(solver)

    sweep_arrays = [RANGES[var] for var in SWEEP_VARS]
    all_points = list(itertools.product(*sweep_arrays))
    
    my_points = chunk(all_points, rank, size)

    my_results = sweep(my_points, lattice, solver, observables, rank)
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        results = {k: v for r in all_results for k, v in r.items()}
        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(results, filepath)
        print(f"Successfully saved data to {filepath}")


if __name__ == "__main__":
    main()
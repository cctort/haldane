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

DATA_DIR = Path('./data/disorder')
RESULTS_FILE = 'sweep.h5'
REGISTRY_FILE = 'configs.json'

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
    'N_SAMPLES': 15,
    'W0': 0.0,
    'W1': 2.0,
    'W2': 0.0
}
for key, value in CONFIG.items():
    setattr(config, key, value)

SWEEP_VARS = ('U', 'V')
FIXED_VARS = {'delta': 0.0}

RANGES = {
    'delta': np.linspace(0, 4, 14),
    'U': np.linspace(0, 12.5, 14),
    'V': np.linspace(0, 4, 14)
}

# Possible observables: 'E_gs', 'gap', 'cdw', 'sdw', 'resta', 'chern'
OBS = ['chern']
REPLACE_OBS = False


def disorder_config(lattice):
    site_dis_list = []
    bond_dis_list = []
    
    for _ in range(config.N_SAMPLES):
        site_dis_list.append((np.random.rand(config.NUM_SITES) - 0.5) * config.W0)
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


def sweep(points, lattice, solver, observables, rank, v0=None):
    results = {}
    calc_gap = 'gap' in OBS

    for pt in points:
        t0 = time.time()
        
        delta, U, V = pt  
        
        site_dis_list, bond_dis_list = disorder_config(lattice)
        sample_data = {obs_key: [] for obs_key in OBS}

        for s in range(config.N_SAMPLES):
            res = solver.ground_state(
                delta, U, V, 
                gap=calc_gap, 
                v0=v0,
                site_dis=site_dis_list[s],
                bond_dis=bond_dis_list[s]
            )

            E, psi = res[0], res[1]
            v0 = psi  

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
                sample_data['chern'].append(observables.chern_number(delta, U, V, grid=config.N_FLUX, v0=None,
                                                                     site_dis=site_dis_list[s], bond_dis=bond_dis_list[s]))

        pt_data = {}
        for key, vals in sample_data.items():
            pt_data[key] = np.mean(vals)
            pt_data[f"{key}_std"] = np.std(vals)

        results[(delta, U, V)] = pt_data

        d_vals = {'delta': delta, 'U': U, 'V': V}
        sweep_str = ", ".join([f"{var}={d_vals[var]:.4f}" for var in SWEEP_VARS])
        print(f"rank {rank}: ({sweep_str}) point done in {time.time()-t0:.1f}s", flush=True)

    return results, v0


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
                    grp.attrs.pop(f"{key}_std", None)

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


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    total_t0 = time.time()

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

    if size == 1:
        results = {}
        v0 = None
        for traj in trajectories:
            traj_results, v0 = sweep(traj, lattice, solver, observables, rank=0, v0=v0)
            results.update(traj_results)
        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(results, filepath)
        print(f"Successfully saved data to {filepath}")
        print(f"Total time elapsed: {time.time() - total_t0:.1f}s", flush=True)
        return

    if rank == 0:
        results = {}
        next_idx = 0
        active_workers = size - 1

        for worker in range(1, size):
            if next_idx < len(trajectories):
                comm.send(trajectories[next_idx], dest=worker, tag=11)
                next_idx += 1
            else:
                comm.send(None, dest=worker, tag=99)
                active_workers -= 1

        while active_workers > 0:
            status = MPI.Status()
            worker_results = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()
            
            results.update(worker_results)

            if next_idx < len(trajectories):
                comm.send(trajectories[next_idx], dest=worker, tag=11)
                next_idx += 1
            else:
                comm.send(None, dest=worker, tag=99)
                active_workers -= 1

        filepath = get_or_create_hdf5_file(DATA_DIR, RESULTS_FILE, current_config)
        save(results, filepath)
        print(f"Successfully saved data to {filepath}")
        print(f"Total time elapsed: {time.time() - total_t0:.1f}s", flush=True)

    else:
        v0 = None
        while True:
            status = MPI.Status()
            traj = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            
            if status.Get_tag() == 99:
                break
            
            traj_results, v0 = sweep(traj, lattice, solver, observables, rank, v0=v0)
            comm.send(traj_results, dest=0, tag=0)


if __name__ == "__main__":
    main()
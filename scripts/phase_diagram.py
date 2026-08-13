"""
Main workflow for computing phase diagrams.

Reproduces Fig. 1(a) and Fig. 1(b) from the paper:

- Fig. 1(a): (Δ, U) phase diagram at V = 0
- Fig. 1(b): (U, V) phase diagram at Δ = 0
"""

import resource
import numpy as np
import h5py
import os
from typing import Dict, Tuple
from pathlib import Path
import time
from mpi4py import MPI

from ed.config import (
    DELTA_RANGE, U_RANGE, V_RANGE, DATA_DIR, RESULTS_FILE, VERBOSE, print_config
)
from ed.lattice import get_lattice
from ed.hamiltonian import HaldaneHubbardHamiltonian
from ed.ed_solver import ExactDiagonalizationSolver
from ed.observables import ObservableCalculator


class PhaseDiagramComputer:

    def __init__(self, output_dir: str = DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.hamiltonian = HaldaneHubbardHamiltonian()
        self.solver = ExactDiagonalizationSolver(self.hamiltonian)
        self.observables = ObservableCalculator(
            self.hamiltonian.basis,
            get_lattice(),
            self.solver
        )

        self.results: Dict[Tuple[float, float, float], Dict] = {}

    def compute_phase_diagram_delta_u(
        self,
        delta_range: np.ndarray = None,
        u_range: np.ndarray = None,
        v: float = 0.0,
        observables: list[str] = None
    ) -> Dict:

        if delta_range is None:
            delta_range = DELTA_RANGE
        if u_range is None:
            u_range = U_RANGE
        if observables is None:
            observables = ['cdw', 'sdw', 'chern']

        results = {}
        n_total = len(delta_range) * len(u_range)
        count = 0

        if self.rank == 0:
            print(f"\n{'='*70}")
            print(f"Computing (Δ, U) phase diagram at V = {v}")
            print(f"  Δ range: {delta_range}")
            print(f"  U range: {u_range}")
            print(f"  Total points: {n_total}")
            print(f"{'='*70}\n")

        points = [(delta, u) for delta in delta_range for u in u_range]
        local_points = points[self.rank::self.size]

        t_start = time.time()

        for delta, u in local_points:
            count += 1
            key = (delta, u, v)

            if VERBOSE:
                print(f"[{count:3d}/{len(local_points)}]", end=" ")

            E_gs, psi_gs = self.solver.solve(delta, u, v)

            results[key] = {'E_gs': E_gs}

            if 'cdw' in observables:
                results[key]['cdw'] = self.observables.compute_cdw(psi_gs)

            if 'sdw' in observables:
                results[key]['sdw'] = self.observables.compute_sdw_squared(psi_gs)

            if 'chern' in observables:
                try:
                    chern = self.observables.compute_chern_number(delta, u, v)
                    results[key]['chern'] = chern
                    results[key]['chern_int'] = int(np.round(chern))
                except Exception as e:
                    if VERBOSE:
                        print(f"\nWarning computing Chern number: {e}")

        t_elapsed = time.time() - t_start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        print(
            f"rank {self.rank}: [{count:3d}/{len(local_points)}] "
            f"in {t_elapsed:.2f} s, RSS={rss:.1f} MB",
            end=" "
        )

        all_results = self.comm.gather(results, root=0)

        if self.rank == 0:
            results = {}
            for rank_results in all_results:
                results.update(rank_results)

            self.results.update(results)
            self._save_results(self.results)
            return results

        return {}

    def compute_phase_diagram_u_v(
        self,
        u_range: np.ndarray = None,
        v_range: np.ndarray = None,
        delta: float = 0.0,
        observables: list[str] = None
    ) -> Dict:

        if u_range is None:
            u_range = U_RANGE
        if v_range is None:
            v_range = V_RANGE
        if observables is None:
            observables = ['cdw', 'sdw', 'chern']

        results = {}
        n_total = len(u_range) * len(v_range)
        count = 0

        if self.rank == 0:
            print(f"\n{'='*70}")
            print(f"Computing (U, V) phase diagram at Δ = {delta}")
            print(f"  U range: {u_range}")
            print(f"  V range: {v_range}")
            print(f"  Total points: {n_total}")
            print(f"{'='*70}\n")

        points = [(u, v) for u in u_range for v in v_range]
        local_points = points[self.rank::self.size]

        t_start = time.time()

        for u, v in local_points:
            count += 1
            key = (delta, u, v)

            if VERBOSE:
                print(f"[{count:3d}/{len(local_points)}]", end=" ")

            E_gs, psi_gs = self.solver.solve(delta, u, v)

            results[key] = {'E_gs': E_gs}

            if 'cdw' in observables:
                results[key]['cdw'] = self.observables.compute_cdw(psi_gs)

            if 'sdw' in observables:
                results[key]['sdw'] = self.observables.compute_sdw_squared(psi_gs)

            if 'chern' in observables:
                try:
                    chern = self.observables.compute_chern_number(delta, u, v)
                    results[key]['chern'] = chern
                    results[key]['chern_int'] = int(np.round(chern))
                except Exception as e:
                    if VERBOSE:
                        print(f"\nWarning computing Chern number: {e}")

        t_elapsed = time.time() - t_start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        print(
            f"rank {self.rank}: [{count:3d}/{len(local_points)}] "
            f"in {t_elapsed:.2f} s, RSS={rss:.1f} MB",
            end=" "
        )

        all_results = self.comm.gather(results, root=0)

        if self.rank == 0:
            results = {}

            for rank_results in all_results:
                results.update(rank_results)

            self.results.update(results)
            self._save_results(self.results)
            return results

        return {}

    def _save_results(self, results: Dict):
        output_file = self.output_dir / RESULTS_FILE

        with h5py.File(output_file, 'w') as f:
            for (delta, u, v), data in results.items():
                group_name = f"delta_{delta:.3f}_U_{u:.3f}_V_{v:.3f}"
                grp = f.create_group(group_name)

                for key, val in data.items():
                    if isinstance(val, (int, float, np.number)):
                        grp.attrs[key] = val

        if VERBOSE:
            print(f"Saved results to {output_file}")

    def print_results_summary(self):
        if not self.results:
            return

        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(
            f"{'Δ':<8} {'U':<8} {'V':<8} {'E_gs':<12} "
            f"{'CDW':<10} {'SDW':<10} {'C':<4}"
        )
        print("-" * 70)

        for (delta, u, v), data in sorted(self.results.items()):
            e = f"{data['E_gs']:<12.6f}"
            cdw = f"{data['cdw']:<10.4f}" if 'cdw' in data else "-"
            sdw = f"{data['sdw']:<10.4f}" if 'sdw' in data else "-"
            chern = f"{data['chern_int']:<4d}" if 'chern_int' in data else "-"

            print(
                f"{delta:<8.3f} {u:<8.3f} {v:<8.3f} "
                f"{e} {cdw} {sdw} {chern}"
            )


def main():
    computer = PhaseDiagramComputer()

    if computer.rank == 0:
        print_config()
        os.makedirs(DATA_DIR, exist_ok=True)
        print("\nComputing Fig. 1(a): (Δ, U) phase diagram at V = 0")

    results_a = computer.compute_phase_diagram_delta_u(
        v=0.0,
        observables=['cdw', 'sdw']
    )

    results_b = computer.compute_phase_diagram_u_v(
        delta=0.0,
        observables=['cdw', 'sdw']
    )

    computer.comm.Barrier()

    if computer.rank == 0:
        computer.print_results_summary()

        output_file = Path(DATA_DIR) / RESULTS_FILE
        print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
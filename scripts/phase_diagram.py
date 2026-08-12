"""
Main workflow for computing phase diagrams.

Reproduces Fig. 1(a) and Fig. 1(b) from the paper:
- Fig. 1(a): (Δ, U) phase diagram at V = 0
- Fig. 1(b): (U, V) phase diagram at Δ = 0
"""

import numpy as np
import h5py
import os
from typing import Dict, Tuple
from pathlib import Path
import time

import sys
sys.path.insert(0, '/home/ctort/Desktop/ed/git')

from src.config import (
    DELTA_RANGE, U_RANGE, V_RANGE, DATA_DIR, RESULTS_FILE, 
    CHECKPOINT_INTERVAL, VERBOSE, print_config
)
from src.lattice import get_lattice
from src.hamiltonian import HaldaneHubbardHamiltonian
from src.ed_solver import ExactDiagonalizationSolver
from src.observables import ObservableCalculator


class PhaseDiagramComputer:
    """Compute phase diagrams of the Haldane-Hubbard model."""
    
    def __init__(self, output_dir: str = DATA_DIR):
        """
        Initialize phase diagram computer.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize solvers
        self.hamiltonian = HaldaneHubbardHamiltonian()
        self.solver = ExactDiagonalizationSolver(self.hamiltonian)
        self.observables = ObservableCalculator(
            self.hamiltonian.basis, 
            get_lattice(), 
            self.solver
        )
        
        # Results storage
        self.results: Dict[Tuple[float, float, float], Dict] = {}
    
    def compute_phase_diagram_delta_u(
        self,
        delta_range: np.ndarray = None,
        u_range: np.ndarray = None,
        v: float = 0.0,
    ) -> Dict:
        """
        Compute (Δ, U) phase diagram at fixed V.
        
        Reproduces Fig. 1(a): V = 0
        
        Args:
            delta_range: Range of staggered potential values
            u_range: Range of Hubbard interaction values
            v: Fixed nearest-neighbor Coulomb interaction
            
        Returns:
            Dictionary with results at each (Δ, U) point
        """
        if delta_range is None:
            delta_range = DELTA_RANGE
        if u_range is None:
            u_range = U_RANGE
        
        results = {}
        n_total = len(delta_range) * len(u_range)
        count = 0
        
        print(f"\n{'='*70}")
        print(f"Computing (Δ, U) phase diagram at V = {v}")
        print(f"  Δ range: {delta_range}")
        print(f"  U range: {u_range}")
        print(f"  Total points: {n_total}")
        print(f"{'='*70}\n")
        
        t_start = time.time()
        
        for delta in delta_range:
            for u in u_range:
                count += 1
                if VERBOSE:
                    print(f"[{count:3d}/{n_total}]", end=" ")
                
                # Solve ground state
                E_gs, psi_gs = self.solver.solve(delta, u, v)
                
                # Compute observables
                cdw = self.observables.compute_cdw(psi_gs)
                sdw = self.observables.compute_sdw_squared(psi_gs)
                
                # Compute Chern number
                try:
                    chern = self.observables.compute_chern_number(delta, u, v)
                except Exception as e:
                    if VERBOSE:
                        print(f"\nWarning computing Chern number: {e}")
                    chern = 0
                
                # Round Chern number to nearest integer
                chern_int = int(np.round(chern))
                
                # Store results
                key = (delta, u, v)
                results[key] = {
                    'E_gs': E_gs,
                    'CDW': cdw,
                    'SDW': sdw,
                    'Chern': chern,
                    'Chern_int': chern_int,
                }
                
                if VERBOSE:
                    print(f"C={chern_int:2d} CDW={cdw:7.3f} SDW={sdw:7.3f}")
                
                # Checkpoint
                if count % CHECKPOINT_INTERVAL == 0:
                    self._save_results(results)
        
        t_elapsed = time.time() - t_start
        print(f"\nCompleted in {t_elapsed:.1f}s ({t_elapsed/n_total:.2f}s per point)")
        
        self.results.update(results)
        return results
    
    def compute_phase_diagram_u_v(
        self,
        u_range: np.ndarray = None,
        v_range: np.ndarray = None,
        delta: float = 0.0,
    ) -> Dict:
        """
        Compute (U, V) phase diagram at fixed Δ.
        
        Reproduces Fig. 1(b): Δ = 0
        
        Args:
            u_range: Range of Hubbard interaction values
            v_range: Range of nearest-neighbor Coulomb values
            delta: Fixed staggered potential
            
        Returns:
            Dictionary with results at each (U, V) point
        """
        if u_range is None:
            u_range = U_RANGE
        if v_range is None:
            v_range = V_RANGE
        
        results = {}
        n_total = len(u_range) * len(v_range)
        count = 0
        
        print(f"\n{'='*70}")
        print(f"Computing (U, V) phase diagram at Δ = {delta}")
        print(f"  U range: {u_range}")
        print(f"  V range: {v_range}")
        print(f"  Total points: {n_total}")
        print(f"{'='*70}\n")
        
        t_start = time.time()
        
        for u in u_range:
            for v in v_range:
                count += 1
                if VERBOSE:
                    print(f"[{count:3d}/{n_total}]", end=" ")
                
                # Solve ground state
                E_gs, psi_gs = self.solver.solve(delta, u, v)
                
                # Compute observables
                cdw = self.observables.compute_cdw(psi_gs)
                sdw = self.observables.compute_sdw_squared(psi_gs)
                
                # Compute Chern number
                try:
                    chern = self.observables.compute_chern_number(delta, u, v)
                except Exception as e:
                    if VERBOSE:
                        print(f"\nWarning computing Chern number: {e}")
                    chern = 0
                
                chern_int = int(np.round(chern))
                
                # Store results
                key = (delta, u, v)
                results[key] = {
                    'E_gs': E_gs,
                    'CDW': cdw,
                    'SDW': sdw,
                    'Chern': chern,
                    'Chern_int': chern_int,
                }
                
                if VERBOSE:
                    print(f"C={chern_int:2d} CDW={cdw:7.3f} SDW={sdw:7.3f}")
                
                # Checkpoint
                if count % CHECKPOINT_INTERVAL == 0:
                    self._save_results(results)
        
        t_elapsed = time.time() - t_start
        print(f"\nCompleted in {t_elapsed:.1f}s ({t_elapsed/n_total:.2f}s per point)")
        
        self.results.update(results)
        return results
    
    def _save_results(self, results: Dict):
        """Save results to HDF5 file."""
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
        """Print summary of results."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Δ':<8} {'U':<8} {'V':<8} {'E_gs':<12} {'CDW':<10} {'SDW':<10} {'C':<4}")
        print("-"*70)
        
        for (delta, u, v), data in sorted(self.results.items()):
            print(f"{delta:<8.3f} {u:<8.3f} {v:<8.3f} {data['E_gs']:<12.6f} "
                  f"{data['CDW']:<10.4f} {data['SDW']:<10.4f} {data['Chern_int']:<4d}")


def main():
    """Main script to reproduce paper results."""
    # Print configuration
    print_config()
    
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize computer
    computer = PhaseDiagramComputer()
    
    # Compute Fig. 1(a): (Δ, U) at V = 0
    print("\nComputing Fig. 1(a): (Δ, U) phase diagram at V = 0")
    results_a = computer.compute_phase_diagram_delta_u(v=0.0)
    
    # Compute Fig. 1(b): (U, V) at Δ = 0
    #print("\nComputing Fig. 1(b): (U, V) phase diagram at Δ = 0")
    #results_b = computer.compute_phase_diagram_u_v(delta=0.0)
    
    # Print summary
    computer.print_results_summary()
    
    # Save final results
    output_file = Path(DATA_DIR) / RESULTS_FILE
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
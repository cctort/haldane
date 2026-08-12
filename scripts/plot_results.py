"""
Plot phase diagrams and observables.

Reproduces figures similar to those in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.config import DATA_DIR


class PhaseDiagramPlotter:
    """Visualize phase diagram results."""
    
    def __init__(self, results_file: str = None):
        """
        Initialize plotter.
        
        Args:
            results_file: Path to HDF5 results file
        """
        if results_file is None:
            results_file = Path(DATA_DIR) / "phase_diagram.h5"
        
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load results from HDF5 file."""
        if not self.results_file.exists():
            print(f"Warning: Results file not found at {self.results_file}")
            return {}
        
        results = {}
        with h5py.File(self.results_file, 'r') as f:
            for key in f.keys():
                group = f[key]
                data = {attr: group.attrs[attr] for attr in group.attrs}
                results[key] = data
        
        return results
    
    def extract_phase_diagram_delta_u(self, v: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract (Δ, U) phase diagram data.
        
        Returns:
            (delta_values, u_values, chern_grid)
        """
        if not self.results:
            print("No results loaded.")
            return None, None, None
        
        # Collect all points for this V
        points = {}
        for key, data in self.results.items():
            # Parse key: delta_X.XXX_U_X.XXX_V_X.XXX
            parts = key.split('_')
            try:
                delta = float(parts[1])
                u = float(parts[3])
                v_val = float(parts[5])
                
                if abs(v_val - v) < 0.01:  # Match V
                    points[(delta, u)] = data['Chern_int']
            except (IndexError, ValueError):
                continue
        
        if not points:
            print(f"No points found for V={v}")
            return None, None, None
        
        # Create grid
        deltas = sorted(set(d for d, u in points.keys()))
        us = sorted(set(u for d, u in points.keys()))
        
        chern_grid = np.zeros((len(deltas), len(us)))
        for i, delta in enumerate(deltas):
            for j, u in enumerate(us):
                chern_grid[i, j] = points.get((delta, u), np.nan)
        
        return np.array(deltas), np.array(us), chern_grid
    
    def extract_phase_diagram_u_v(self, delta: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract (U, V) phase diagram data.
        
        Returns:
            (u_values, v_values, chern_grid)
        """
        if not self.results:
            print("No results loaded.")
            return None, None, None
        
        # Collect all points for this Δ
        points = {}
        for key, data in self.results.items():
            parts = key.split('_')
            try:
                delta_val = float(parts[1])
                u = float(parts[3])
                v = float(parts[5])
                
                if abs(delta_val - delta) < 0.01:  # Match Δ
                    points[(u, v)] = data['Chern_int']
            except (IndexError, ValueError):
                continue
        
        if not points:
            print(f"No points found for Δ={delta}")
            return None, None, None
        
        # Create grid
        us = sorted(set(u for u, v in points.keys()))
        vs = sorted(set(v for u, v in points.keys()))
        
        chern_grid = np.zeros((len(us), len(vs)))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                chern_grid[i, j] = points.get((u, v), np.nan)
        
        return np.array(us), np.array(vs), chern_grid
    
    def plot_phase_diagram_delta_u(self, v: float = 0.0, figsize: Tuple = (10, 8)):
        """
        Plot (Δ, U) phase diagram at fixed V.
        
        Reproduces Fig. 1(a)
        """
        deltas, us, chern_grid = self.extract_phase_diagram_delta_u(v=v)
        
        if chern_grid is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap for Chern number
        colors = ['white', 'yellow', 'lightgreen', 'lightblue', 'lightcoral', 'black']
        chern_values = [0, 1, 2, 3, 4, 5]
        
        # Plot as image
        im = ax.imshow(
            chern_grid.T,
            origin='lower',
            extent=[deltas[0], deltas[-1], us[0], us[-1]],
            cmap='tab10',
            aspect='auto',
            interpolation='nearest'
        )
        
        # Add contour lines for phase boundaries
        contours = ax.contour(deltas, us, chern_grid.T, levels=[0.5, 1.5, 2.5], colors='black', linewidths=0.5)
        
        ax.set_xlabel('Staggered potential Δ/t', fontsize=12)
        ax.set_ylabel('Hubbard interaction U/t', fontsize=12)
        ax.set_title(f'Phase diagram: Chern number C(Δ, U) at V={v}', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax, label='Chern number C')
        
        fig.tight_layout()
        return fig
    
    def plot_phase_diagram_u_v(self, delta: float = 0.0, figsize: Tuple = (10, 8)):
        """
        Plot (U, V) phase diagram at fixed Δ.
        
        Reproduces Fig. 1(b)
        """
        us, vs, chern_grid = self.extract_phase_diagram_u_v(delta=delta)
        
        if chern_grid is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot as image
        im = ax.imshow(
            chern_grid.T,
            origin='lower',
            extent=[us[0], us[-1], vs[0], vs[-1]],
            cmap='tab10',
            aspect='auto',
            interpolation='nearest'
        )
        
        # Add contour lines
        contours = ax.contour(us, vs, chern_grid.T, levels=[0.5, 1.5, 2.5], colors='black', linewidths=0.5)
        
        ax.set_xlabel('Hubbard interaction U/t', fontsize=12)
        ax.set_ylabel('Coulomb interaction V/t', fontsize=12)
        ax.set_title(f'Phase diagram: Chern number C(U, V) at Δ={delta}', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax, label='Chern number C')
        
        fig.tight_layout()
        return fig
    
    def plot_structure_factors(self, delta: float, u: float, v: float = 0.0, figsize: Tuple = (12, 5)):
        """
        Plot CDW and SDW structure factors around a point.
        
        Useful for understanding phases near a specific parameter set.
        """
        # TODO: Extract and plot CDW/SDW for a grid around (delta, u)
        pass
    
    def save_figures(self, output_dir: str = "./figures"):
        """Save all phase diagrams to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Fig 1(a)
        fig_a = self.plot_phase_diagram_delta_u(v=0.0)
        if fig_a is not None:
            fig_a.savefig(f"{output_dir}/fig_1a_phase_diagram_delta_u.png", dpi=150)
            print(f"Saved: {output_dir}/fig_1a_phase_diagram_delta_u.png")
        
        # Fig 1(b)
        fig_b = self.plot_phase_diagram_u_v(delta=0.0)
        if fig_b is not None:
            fig_b.savefig(f"{output_dir}/fig_1b_phase_diagram_u_v.png", dpi=150)
            print(f"Saved: {output_dir}/fig_1b_phase_diagram_u_v.png")


def main():
    """Plot results from phase diagram calculation."""
    plotter = PhaseDiagramPlotter()
    
    # Plot both phase diagrams
    print("Plotting (Δ, U) phase diagram at V=0...")
    fig_a = plotter.plot_phase_diagram_delta_u(v=0.0)
    if fig_a is not None:
        plt.show()
    
    print("\nPlotting (U, V) phase diagram at Δ=0...")
    fig_b = plotter.plot_phase_diagram_u_v(delta=0.0)
    if fig_b is not None:
        plt.show()
    
    # Save figures
    print("\nSaving figures...")
    plotter.save_figures()


if __name__ == "__main__":
    main()

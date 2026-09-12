import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_FILE = "phase_diagram.h5"

IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})


class PhaseDiagramPlotter:
    def __init__(self, results_file=None):
        self.results_file = Path(results_file) if results_file else DATA_DIR / RESULTS_FILE
        self.points = self._load_points()

    def _load_points(self):
        if not self.results_file.exists():
            return []

        points = []
        with h5py.File(self.results_file, "r") as f:
            for group in f.values():
                if not all(k in group.attrs for k in ("delta", "U", "V")):
                    continue

                chern = group.attrs.get("chern")
                chern_val = float(chern) if chern is not None else None

                points.append({
                    "delta": float(group.attrs["delta"]),
                    "u": float(group.attrs["U"]),
                    "v": float(group.attrs["V"]),
                    "cdw": group.attrs.get("cdw"),
                    "sdw": group.attrs.get("sdw"),
                    "gap": group.attrs.get("gap"),
                    "chern": chern_val,
                })

        return points

    def _filter_points(self, fixed_var, fixed_val):
        return [p for p in self.points if np.isclose(p[fixed_var], fixed_val)]

    def _nearest_grid(self, points, x_key, y_key, value_key, n=400):
        x = np.array([p[x_key] for p in points])
        y = np.array([p[y_key] for p in points])
        vals = np.array([p[value_key] for p in points], dtype=float)

        xg = np.linspace(x.min(), x.max(), n)
        yg = np.linspace(y.min(), y.max(), n)
        X, Y = np.meshgrid(xg, yg)

        d2 = (X[..., None] - x) ** 2 + (Y[..., None] - y) ** 2
        nearest = np.argmin(d2, axis=2)

        return xg, yg, vals[nearest]

    def plot_slice(self, x_key, y_key, fixed_var, fixed_val, value_key, xlabel, ylabel, title, filename, cmap="hot"):
        points = self._filter_points(fixed_var, fixed_val)
        points = [p for p in points if p[value_key] is not None]

        if value_key == "gap":
            points = [p for p in points if p["gap"] > 0]

        if not points:
            return

        values = np.array([p[value_key] for p in points])
        vmin, vmax = np.nanmin(values), np.nanmax(values)

        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.01, 1e-12)
            vmin, vmax = vmin - eps, vmax + eps

        xs, ys, grid = self._nearest_grid(points, x_key, y_key, value_key)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        
        scatter = ax.scatter([p[x_key] for p in points], [p[y_key] for p in points],
                             c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=65, edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(r"$1/\mathrm{Gap}$" if value_key == "gap"
                       else rf"${value_key.upper()}$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_chern_slice(self, x_key, y_key, fixed_var, fixed_val, xlabel, ylabel, title, filename):
        points = self._filter_points(fixed_var, fixed_val)
        points = [p for p in points if p["chern"] is not None]

        if not points:
            return

        chern_vals = [p["chern"] for p in points]
        lo = int(round(min(chern_vals)))
        hi = int(round(max(chern_vals)))

        if lo == hi:
            lo -= 1
            hi += 1

        xs, ys, grid = self._nearest_grid(points, x_key, y_key, "chern")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_classes = max(1, hi - lo + 1)
        cmap = plt.get_cmap("viridis", num_classes)
        bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)

        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=cmap, norm=norm, zorder=1)

        scatter = ax.scatter([p[x_key] for p in points], [p[y_key] for p in points],
                             c=chern_vals, cmap=cmap, norm=norm, s=65,
                             edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)

        ticks = list(range(lo, hi + 1))
        cbar = fig.colorbar(scatter, ax=ax, ticks=ticks, boundaries=bounds)
        cbar.set_label(r"$\mathrm{Chern\ number}$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_all(self):
        self.plot_chern_slice("u", "delta", "v", 0.0, r"$U/t$", r"$\Delta/t$",
                              r"$\mathrm{Chern\ number}:\ \Delta/t\ \mathrm{vs.}\ U/t,\ V/t=0$",
                              "chern_delta_u.png")

        self.plot_chern_slice("u", "v", "delta", 0.0, r"$U/t$", r"$V/t$",
                              r"$\mathrm{Chern\ number}:\ V/t\ \mathrm{vs.}\ U/t,\ \Delta/t=0$",
                              "chern_v_u.png")

        for key, label in (("cdw", "CDW"), ("sdw", "SDW"), ("gap", "Gap")):
            self.plot_slice("u", "delta", "v", 0.0, key, r"$U/t$", r"$\Delta/t$",
                            rf"$\mathrm{{{label}}}:\ \Delta/t\ \mathrm{{vs.}}\ U/t,\ V/t=0$",
                            f"{key}_delta_u.png")

            self.plot_slice("u", "v", "delta", 0.0, key, r"$U/t$", r"$V/t$",
                            rf"$\mathrm{{{label}}}:\ V/t\ \mathrm{{vs.}}\ U/t,\ \Delta/t=0$",
                            f"{key}_v_u.png")


def main():
    PhaseDiagramPlotter().plot_all()


if __name__ == "__main__":
    main()
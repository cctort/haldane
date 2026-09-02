import sys
import re
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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

KEY_RE = re.compile(
    r"delta_([-+]?\d*\.?\d+)_U_([-+]?\d*\.?\d+)_V_([-+]?\d+\.?\d*)"
)


class PhaseDiagramPlotter:
    def __init__(self, results_file=None):
        self.results_file = Path(results_file) if results_file else DATA_DIR / RESULTS_FILE
        self.points = self._load_points()

    def _load_points(self):
        if not self.results_file.exists():
            return []

        points = []

        with h5py.File(self.results_file, "r") as f:
            for key, group in f.items():
                m = KEY_RE.match(key)
                if not m:
                    continue

                delta, u, v = (float(x) for x in m.groups())

                chern = group.attrs.get("chern")
                chern_int, chern_ok = None, True

                if chern is not None:
                    chern_int = int(round(float(chern)))
                    chern_ok = abs(float(chern) - chern_int) < 0.1

                points.append({
                    "delta": delta,
                    "u": u,
                    "v": v,
                    "cdw": group.attrs.get("cdw"),
                    "sdw": group.attrs.get("sdw"),
                    "gap": group.attrs.get("gap"),
                    "chern": chern_int if chern_ok else None,
                })

        return points

    def _delta_u(self, v):
        return [p for p in self.points if np.isclose(p["v"], v)]

    def _u_v(self, delta):
        return [p for p in self.points if np.isclose(p["delta"], delta)]

    def _nearest_grid(self, points, y_key, value_key, n=400):
        x = np.array([p["u"] for p in points])
        y = np.array([p[y_key] for p in points])
        vals = np.array([p[value_key] for p in points], dtype=float)

        xg = np.linspace(x.min(), x.max(), n)
        yg = np.linspace(y.min(), y.max(), n)
        X, Y = np.meshgrid(xg, yg)

        d2 = (X[..., None] - x) ** 2 + (Y[..., None] - y) ** 2
        nearest = np.argmin(d2, axis=2)

        return xg, yg, vals[nearest]

    def _plot_observable(self, points, y_key, value_key, ylabel, title, filename, cmap="hot"):
        points = [p for p in points if p[value_key] is not None]

        if value_key == "gap":
            points = [p for p in points if p["gap"] > 0]

        if not points:
            return
        
        else:
            values = np.array([p[value_key] for p in points])
            plot_key = value_key

        vmin, vmax = np.nanmin(values), np.nanmax(values)

        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.01, 1e-12)
            vmin, vmax = vmin - eps, vmax + eps

        xs, ys, grid = self._nearest_grid(points, y_key, plot_key)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        scatter = ax.scatter([p["u"] for p in points], [p[y_key] for p in points],
                             c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=65, edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(r"$U/t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(r"$1/\mathrm{Gap}$" if value_key == "gap"
                       else rf"${value_key.upper()}$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _chern_plot(self, points, y_key, ylabel, title, filename):
        points = [p for p in points if p["chern"] is not None]

        if not points:
            return

        chern_vals = [p["chern"] for p in points]
        lo, hi = min(chern_vals), max(chern_vals)

        levels = list(range(lo, hi + 1))
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(levels))))
        norm = BoundaryNorm([c - 0.5 for c in levels] + [levels[-1] + 0.5], cmap.N)

        xs, ys, grid = self._nearest_grid(points, y_key, "chern")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=cmap, norm=norm, zorder=1)
        scatter = ax.scatter([p["u"] for p in points], [p[y_key] for p in points],
                             c=chern_vals, cmap=cmap, norm=norm, s=65,
                             edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(r"$U/t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)

        cbar = fig.colorbar(scatter, ax=ax, ticks=levels)
        cbar.set_label(r"$\mathrm{Chern\ number}$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        fig.tight_layout()
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_all(self):
        delta_u = self._delta_u(0.0)
        u_v = self._u_v(0.0)

        self._chern_plot(delta_u, "delta", r"$\Delta/t$",
                         r"$\mathrm{Chern\ number}:\ \Delta/t\ \mathrm{vs.}\ U/t,\ V/t=0$",
                         "chern_delta_u.png")

        self._chern_plot(u_v, "v", r"$V/t$",
                         r"$\mathrm{Chern\ number}:\ V/t\ \mathrm{vs.}\ U/t,\ \Delta/t=0$",
                         "chern_v_u.png")

        for key, label in (("cdw", "CDW"), ("sdw", "SDW"), ("gap", "Gap")):
            self._plot_observable(delta_u, "delta", key, r"$\Delta/t$",
                                  rf"$\mathrm{{{label}}}:\ \Delta/t\ \mathrm{{vs.}}\ U/t,\ V/t=0$",
                                  f"{key}_delta_u.png")

            self._plot_observable(u_v, "v", key, r"$V/t$",
                                  rf"$\mathrm{{{label}}}:\ V/t\ \mathrm{{vs.}}\ U/t,\ \Delta/t=0$",
                                  f"{key}_v_u.png")


def main():
    PhaseDiagramPlotter().plot_all()


if __name__ == "__main__":
    main()
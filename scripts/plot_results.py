import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re
import numpy as np
import matplotlib.pyplot as plt
import h5py

from ed.config import DATA_DIR
from matplotlib.colors import ListedColormap, BoundaryNorm

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
        if results_file is None:
            results_file = Path(DATA_DIR) / "phase_diagram.h5"

        self.results_file = Path(results_file)
        self.results = self._load_results()

    def _load_results(self):
        if not self.results_file.exists():
            return {}

        results = {}

        with h5py.File(self.results_file, "r") as f:
            for key in f.keys():
                group = f[key]
                results[key] = {
                    name: group.attrs[name]
                    for name in group.attrs
                }

        return results

    def _parse_key(self, key):
        patterns = [
            r"[Dd]elta[_=]?([-+]?\d*\.?\d+).*?[Uu][_=]?([-+]?\d*\.?\d+).*?[Vv][_=]?([-+]?\d*\.?\d+)",
            r"[Dd][_=]?([-+]?\d*\.?\d+).*?[Uu][_=]?([-+]?\d*\.?\d+).*?[Vv][_=]?([-+]?\d*\.?\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, key)

            if match:
                return tuple(float(x) for x in match.groups())

        return None

    def _attr(self, data, names):
        for name in names:
            if name in data:
                return float(data[name])

        return None

    def _get_points(self):
        points = []

        for key, data in self.results.items():
            parsed = self._parse_key(key)

            if parsed is not None:
                delta, u, v = parsed

            else:
                delta = self._attr(data, ["Delta", "delta", "Delta/t", "delta/t"])
                u = self._attr(data, ["U", "u", "U/t", "u/t"])
                v = self._attr(data, ["V", "v", "V/t", "v/t"])

                if delta is None or u is None or v is None:
                    continue

            chern = self._attr(data, ["Chern_int", "chern", "Chern"])
            cdw = self._attr(data, ["CDW", "cdw", "CDW_structure_factor", "cdw_structure_factor"])
            sdw = self._attr(data, ["SDW", "sdw", "SDW_structure_factor", "sdw_structure_factor"])

            points.append({
                "delta": delta,
                "u": u,
                "v": v,
                "chern": chern,
                "cdw": cdw,
                "sdw": sdw,
            })

        return points

    def _delta_u(self, v):
        return [
            p for p in self._get_points()
            if np.isclose(p["v"], v)
        ]

    def _u_v(self, delta):
        return [
            p for p in self._get_points()
            if np.isclose(p["delta"], delta)
        ]

    def _make_grid(self, points, y_key, value_key):
        xs = np.array(sorted(set(p["u"] for p in points)))
        ys = np.array(sorted(set(p[y_key] for p in points)))

        grid = np.full(
            (len(ys), len(xs)),
            np.nan
        )

        for p in points:
            i = np.argmin(np.abs(ys - p[y_key]))
            j = np.argmin(np.abs(xs - p["u"]))

            if p[value_key] is not None:
                grid[i, j] = p[value_key]

        return xs, ys, grid

    def _nearest_grid(self, points, y_key, value_key, nx=500, ny=500):
        """
        Construct a dense nearest-neighbour grid directly from the
        actual calculated scatter points.

        Every location is assigned the value of the closest
        calculated point.
        """

        x_points = np.array([
            p["u"] for p in points
        ])

        y_points = np.array([
            p[y_key] for p in points
        ])

        values = np.array([
            p[value_key] for p in points
        ])

        x_grid = np.linspace(
            x_points.min(),
            x_points.max(),
            nx,
        )

        y_grid = np.linspace(
            y_points.min(),
            y_points.max(),
            ny,
        )

        X, Y = np.meshgrid(
            x_grid,
            y_grid,
        )

        distances = (
            (X[..., None] - x_points[None, None, :]) ** 2
            + (Y[..., None] - y_points[None, None, :]) ** 2
        )

        nearest = np.argmin(
            distances,
            axis=2,
        )

        grid = values[nearest]

        return x_grid, y_grid, grid

    def _plot_observable(self, points, y_key, value_key, ylabel, title, filename, cmap="hot", interpolation="nearest"):
        points = [
            p for p in points
            if p[value_key] is not None
        ]

        if not points:
            return

        values = np.array([
            p[value_key]
            for p in points
        ])

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.01, 1e-12)
            vmin -= eps
            vmax += eps

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        if interpolation == "nearest":
            xs, ys, grid = self._nearest_grid(
                points,
                y_key,
                value_key,
            )

            ax.imshow(
                grid,
                origin="lower",
                extent=[
                    xs[0],
                    xs[-1],
                    ys[0],
                    ys[-1],
                ],
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )

        elif interpolation in ("bilinear", "bicubic"):
            xs, ys, grid = self._make_grid(
                points,
                y_key,
                value_key,
            )

            ax.imshow(
                grid,
                origin="lower",
                extent=[
                    xs[0],
                    xs[-1],
                    ys[0],
                    ys[-1],
                ],
                aspect="auto",
                interpolation=interpolation,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )

        else:
            raise ValueError(
                f"Unknown interpolation method: {interpolation}"
            )

        scatter = ax.scatter(
            [p["u"] for p in points],
            [p[y_key] for p in points],
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=65,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        ax.set_xlabel(
            r"$U/t$"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title,
            pad=12,
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
        )

        cbar.set_label(
            rf"${value_key.upper()}$",
            fontsize=18,
        )

        cbar.ax.tick_params(
            labelsize=16
        )

        fig.tight_layout()

        fig.savefig(
            IMAGES_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _chern_plot(self, points, y_key, ylabel, title, filename):
        points = [
            p for p in points
            if p["chern"] is not None
        ]

        if not points:
            return

        cmap = ListedColormap([
            "purple",
            "green",
            "gold",
        ])

        norm = BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5],
            cmap.N,
        )

        xs, ys, grid = self._nearest_grid(
            points,
            y_key,
            "chern",
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        ax.imshow(
            grid,
            origin="lower",
            extent=[
                xs[0],
                xs[-1],
                ys[0],
                ys[-1],
            ],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            zorder=1,
        )

        scatter = ax.scatter(
            [p["u"] for p in points],
            [p[y_key] for p in points],
            c=[p["chern"] for p in points],
            cmap=cmap,
            norm=norm,
            s=65,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        ax.set_xlabel(
            r"$U/t$"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title,
            pad=12,
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            ticks=[0, 1, 2],
        )

        cbar.set_label(
            r"$\mathrm{Chern\ number}$",
            fontsize=18,
        )

        cbar.ax.tick_params(
            labelsize=16
        )

        fig.tight_layout()

        fig.savefig(
            IMAGES_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    def plot_all(self):
        delta_u = self._delta_u(0.0)
        u_v = self._u_v(0.0)

        self._chern_plot(
            delta_u,
            "delta",
            r"$\Delta/t$",
            r"$\mathrm{Chern\ number}: \Delta/t\ \mathrm{vs.}\ U/t,\quad V/t=0$",
            "chern_delta_u.png",
        )

        self._chern_plot(
            u_v,
            "v",
            r"$V/t$",
            r"$\mathrm{Chern\ number}: V/t\ \mathrm{vs.}\ U/t,\quad \Delta/t=0$",
            "chern_v_u.png",
        )

        self._plot_observable(
            delta_u,
            "delta",
            "cdw",
            r"$\Delta/t$",
            r"$\mathrm{CDW}: \Delta/t\ \mathrm{vs.}\ U/t,\quad V/t=0$",
            "cdw_delta_u.png",
            cmap="hot",
            interpolation="nearest",
        )

        self._plot_observable(
            u_v,
            "v",
            "cdw",
            r"$V/t$",
            r"$\mathrm{CDW}: V/t\ \mathrm{vs.}\ U/t,\quad \Delta/t=0$",
            "cdw_v_u.png",
            cmap="hot",
            interpolation="nearest",
        )

        self._plot_observable(
            delta_u,
            "delta",
            "sdw",
            r"$\Delta/t$",
            r"$\mathrm{SDW}: \Delta/t\ \mathrm{vs.}\ U/t,\quad V/t=0$",
            "sdw_delta_u.png",
            cmap="hot",
            interpolation="nearest",
        )

        self._plot_observable(
            u_v,
            "v",
            "sdw",
            r"$V/t$",
            r"$\mathrm{SDW}: V/t\ \mathrm{vs.}\ U/t,\quad \Delta/t=0$",
            "sdw_v_u.png",
            cmap="hot",
            interpolation="nearest",
        )


def main():
    PhaseDiagramPlotter().plot_all()


if __name__ == "__main__":
    main()
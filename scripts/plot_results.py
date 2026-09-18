import sys
import json
import math
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGISTRY_FILE = "configs.json"

IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

COLOR_LIST = sns.color_palette('colorblind') + sns.color_palette("Set2") + sns.color_palette("Set3")

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


def find_matching_file(target_config, data_dir):
    data_dir = Path(data_dir)
    registry_path = data_dir / REGISTRY_FILE
    if not registry_path.exists():
        return None

    with open(registry_path, 'r') as f:
        try:
            registry = json.load(f)
        except json.JSONDecodeError:
            return None

    def is_match(v_saved, v_target):
        if isinstance(v_saved, (int, float)) and isinstance(v_target, (int, float)):
            return np.isclose(v_saved, v_target, rtol=1e-5, atol=1e-15)
        return v_saved == v_target

    for filename, saved_config in registry.items():
        matches = all(
            k in saved_config and is_match(saved_config[k], v)
            for k, v in target_config.items()
        )
        if matches:
            path = data_dir / filename
            return path if path.exists() else None
            
    return None


def load_points(results_file):
    points = []
    with h5py.File(results_file, "r") as f:
        for group in f.values():
            if not all(k in group.attrs for k in ("delta", "U", "V")):
                continue

            points.append({
                "delta": float(group.attrs["delta"]),
                "u": float(group.attrs["U"]),
                "v": float(group.attrs["V"]),
                "E_gs": group.attrs.get("E_gs"),
                "cdw": group.attrs.get("cdw"),
                "sdw": group.attrs.get("sdw"),
                "gap": group.attrs.get("gap"),
                "chern": group.attrs.get("chern"),
                "resta": group.attrs.get("resta"),
            })

    return points


def get_points(config, data_dir):
    results_file = find_matching_file(config, data_dir)
    if not results_file:
        print(f"No matching file found for config: {config}")
        return []
    return load_points(results_file)


def filter_points(points, fixed_var, fixed_val):
    return [p for p in points if np.isclose(p[fixed_var], fixed_val)]


def crop_points(points, x_key, xlim, y_key=None, ylim=None):
    filtered = points
    if xlim is not None:
        filtered = [p for p in filtered if xlim[0] <= p[x_key] <= xlim[1]]
    if ylim is not None and y_key is not None:
        filtered = [p for p in filtered if ylim[0] <= p[y_key] <= ylim[1]]
    return filtered


def nearest_grid(points, x_key, y_key, value_key, n=400, xlim=None, ylim=None):
    x = np.array([p[x_key] for p in points])
    y = np.array([p[y_key] for p in points])
    vals = np.array([p[value_key] for p in points], dtype=float)

    x0, x1 = xlim if xlim else (x.min(), x.max())
    y0, y1 = ylim if ylim else (y.min(), y.max())

    xg = np.linspace(x0, x1, n)
    yg = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xg, yg)

    d2 = (X[..., None] - x) ** 2 + (Y[..., None] - y) ** 2
    nearest = np.argmin(d2, axis=2)

    return xg, yg, vals[nearest]


def _get_obs_val(p, key):
    val = p.get(key)
    if val is None:
        return None
    if key == "gap":
        return 1.0 / val if not np.isclose(val, 0, atol=1e-4) else np.nan
    return val


def plot_diagram(data_dir, configs, x_key, y_key, fixed_var, fixed_val, value_key, value_label, xlabel, ylabel, titles, filename=None, cmap="hot", xlim=None, ylim=None):
    if isinstance(configs, dict):
        configs = [configs]
    n_configs = len(configs)

    if isinstance(value_key, str):
        value_key = [value_key]
    n_obs = len(value_key)

    if isinstance(value_label, str):
        value_label = [value_label] * n_obs
    elif isinstance(value_label, dict):
        value_label = [value_label.get(k, k) for k in value_key]
    elif isinstance(value_label, list):
        value_label = value_label + [str(value_key[i]) for i in range(len(value_label), n_obs)]

    if isinstance(titles, str):
        titles = [titles] * n_configs

    # Handle the layout rules requested
    if n_configs == 1:
        n_rows = 1
        n_cols = n_obs
    else:
        n_rows = n_obs
        n_cols = n_configs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)

    for c_idx, config in enumerate(configs):
        points = get_points(config, data_dir)
        points = filter_points(points, fixed_var, fixed_val)
        points = crop_points(points, x_key, xlim, y_key, ylim)

        for o_idx, v_key in enumerate(value_key):
            # Select the correct axis depending on the grid shape
            if n_configs == 1:
                ax = axes[0, o_idx]
            else:
                ax = axes[o_idx, c_idx]

            v_label = value_label[o_idx]
            if v_label == "chern":
                v_label = r"$\mathrm{Chern\ number}$"

            if v_key == "gap":
                obs_points = []
                for p in points:
                    val = _get_obs_val(p, "gap")
                    if val is not None and not np.isnan(val) and val > 0:
                        p_copy = p.copy()
                        p_copy["gap"] = val
                        obs_points.append(p_copy)
            else:
                obs_points = [p for p in points if p.get(v_key) is not None]

            if not obs_points:
                print(f'No data fits the chosen parameter constraints for config index {c_idx} ({v_key})')
                ax.axis('off')
                continue

            values = np.array([p[v_key] for p in obs_points])
            xs, ys, grid = nearest_grid(obs_points, x_key, y_key, v_key, xlim=xlim, ylim=ylim)
            scatter_size = 65 / (len(obs_points) / (21**2))

            if v_key == "chern":
                lo = int(round(np.nanmin(values)))
                hi = int(round(np.nanmax(values)))

                if lo == hi:
                    lo -= 1
                    hi += 1

                num_classes = max(1, hi - lo + 1)
                plot_cmap_name = "viridis" if cmap == "hot" else cmap
                plot_cmap = plt.get_cmap(plot_cmap_name, num_classes)
                bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
                norm = BoundaryNorm(bounds, plot_cmap.N)

                ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                          aspect="auto", cmap=plot_cmap, norm=norm, zorder=1)

                scatter = ax.scatter([p[x_key] for p in obs_points], [p[y_key] for p in obs_points],
                                     c=values, cmap=plot_cmap, norm=norm, s=scatter_size,
                                     edgecolors="black", linewidths=0.8, zorder=4)

                ticks = list(range(lo, hi + 1))
                cbar = fig.colorbar(scatter, ax=ax, ticks=ticks, boundaries=bounds)

            elif v_key == "gap":
                vmin, vmax = np.nanmin(values), np.nanmax(values)
                if np.isclose(vmin, vmax):
                    eps = max(abs(vmin) * 0.01, 1e-12)
                    vmin, vmax = vmin - eps, vmax + eps

                norm = LogNorm(vmin=vmin, vmax=vmax)

                ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                          aspect="auto", cmap=cmap, norm=norm, zorder=1)

                scatter = ax.scatter([p[x_key] for p in obs_points], [p[y_key] for p in obs_points],
                                     c=values, cmap=cmap, norm=norm, s=scatter_size,
                                     edgecolors="black", linewidths=0.8, zorder=4)

                cbar = fig.colorbar(scatter, ax=ax)

            else:
                vmin, vmax = np.nanmin(values), np.nanmax(values)

                if np.isclose(vmin, vmax):
                    eps = max(abs(vmin) * 0.01, 1e-12)
                    vmin, vmax = vmin - eps, vmax + eps

                ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                          aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)

                scatter = ax.scatter([p[x_key] for p in obs_points], [p[y_key] for p in obs_points],
                                     c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                                     s=scatter_size, edgecolors="black", linewidths=0.8, zorder=4)

                cbar = fig.colorbar(scatter, ax=ax)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Use the correct title for the top of each row/column dynamically 
            if n_configs == 1 or o_idx == 0:
                ax.set_title(titles[c_idx] if c_idx < len(titles) else titles[-1], pad=plt.rcParams["axes.titlesize"]//2)

            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)

            cbar.set_label(v_label, fontsize=plt.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes


def plot_1d_cut(data_dir, configs, x_key, fixed_vars, value_key, ylabel, xlabel, title, filename=None, colors=None, labels=None, xlim=None, relative_to=None):
    if isinstance(configs, dict):
        configs = [configs]
    n_configs = len(configs)

    if isinstance(value_key, str):
        value_key = [value_key]
    n_plots = len(value_key)

    if colors is None:
        colors = COLOR_LIST
    elif not isinstance(colors, list):
        colors = [colors] * n_configs

    if labels is None:
        labels = [None] * n_configs
    elif isinstance(labels, str):
        labels = [labels] * n_configs
    elif len(labels) != n_configs:
        raise ValueError("Length of labels must match length of configs")

    rows = min(3, n_plots)
    cols = math.ceil(n_plots / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 1 + 2 * rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    if isinstance(ylabel, str):
        ylabel_dict = {obs: ylabel for obs in value_key}
    elif isinstance(ylabel, list):
        ylabel_dict = {value_key[i]: ylabel[i] for i in range(min(n_plots, len(ylabel)))}
    elif isinstance(ylabel, dict):
        ylabel_dict = ylabel
    else:
        ylabel_dict = {obs: str(obs) for obs in value_key}

    if isinstance(title, str):
        title_dict = {obs: title for obs in value_key}
    elif isinstance(title, list):
        title_dict = {value_key[i]: title[i] for i in range(min(n_plots, len(title)))}
    elif isinstance(title, dict):
        title_dict = title
    else:
        title_dict = {obs: str(obs) for obs in value_key}

    indices = list(range(n_configs))
    if relative_to is not None:
        if not (0 <= relative_to < n_configs):
            raise ValueError(f"relative_to index {relative_to} out of range for {n_configs} configs")
        indices.remove(relative_to)
        indices.insert(0, relative_to)

    for plot_idx, obs in enumerate(value_key):
        ax = axes_flat[plot_idx]
        lines_plotted = 0
        ref_dict = {}

        for loop_idx, i in enumerate(indices):
            config = configs[i]
            points = get_points(config, data_dir)
            
            for f_var, f_val in fixed_vars.items():
                points = filter_points(points, f_var, f_val)
                
            points = crop_points(points, x_key, xlim)

            if relative_to is not None and loop_idx == 0:
                points = [p for p in points if _get_obs_val(p, obs) is not None and not np.isnan(_get_obs_val(p, obs))]
                ref_dict = {p[x_key]: _get_obs_val(p, obs) for p in points}
                continue

            if relative_to is not None:
                points = [p for p in points if p[x_key] in ref_dict and _get_obs_val(p, obs) is not None and not np.isnan(_get_obs_val(p, obs))]
            else:
                points = [p for p in points if _get_obs_val(p, obs) is not None and not np.isnan(_get_obs_val(p, obs))]

            if not points:
                continue

            points.sort(key=lambda p: p[x_key])

            x_vals = [p[x_key] for p in points]
            
            if relative_to is not None:
                y_vals = [_get_obs_val(p, obs) - ref_dict[p[x_key]] for p in points]
            else:
                y_vals = [_get_obs_val(p, obs) for p in points]

            current_color = colors[i % len(colors)]
            current_label = labels[i] if i < len(labels) else None

            ax.plot(x_vals, y_vals, '-o', markersize=6, color=current_color, label=current_label)
            lines_plotted += 1

        if lines_plotted > 0:
            if plot_idx == 0: ax.set_title(title_dict.get(obs, str(obs)), pad=plt.rcParams["axes.titlesize"]//2)
            if plot_idx == len(value_key) - 1: ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel_dict.get(obs, str(obs)))
            
            if xlim: 
                ax.set_xlim(xlim)
                
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if any(labels) and plot_idx == 0: ax.legend()
        else:
            ax.axis('off')

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes
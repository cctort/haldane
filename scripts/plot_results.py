import sys
import json
import math
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data/tuning"
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


def find_matching_file(target_config):
    registry_path = DATA_DIR / REGISTRY_FILE
    if not registry_path.exists():
        return None

    with open(registry_path, 'r') as f:
        try:
            registry = json.load(f)
        except json.JSONDecodeError:
            return None

    for filename, saved_config in registry.items():
        matches = all(
            k in saved_config and np.isclose(saved_config[k], v)
            for k, v in target_config.items()
        )
        if matches:
            path = DATA_DIR / filename
            return path if path.exists() else None
            
    return None


def load_points(results_file):
    points = []
    with h5py.File(results_file, "r") as f:
        for group in f.values():
            if not all(k in group.attrs for k in ("delta", "U", "V")):
                continue

            # .get() safely handles attributes missing from the current OBS list
            points.append({
                "delta": float(group.attrs["delta"]),
                "u": float(group.attrs["U"]),
                "v": float(group.attrs["V"]),
                "E_gs": group.attrs.get("E_gs"),
                "cdw": group.attrs.get("cdw"),
                "sdw": group.attrs.get("sdw"),
                "gap": group.attrs.get("gap"),
                "chern": group.attrs.get("chern"),
            })

    return points


def get_points(config):
    results_file = find_matching_file(config)
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


def plot_diagram(configs, x_key, y_key, fixed_var, fixed_val, value_key, value_label, xlabel, ylabel, titles, filename=None, cmap="hot", xlim=None, ylim=None):
    if isinstance(configs, dict):
        configs = [configs]
    if isinstance(titles, str):
        titles = [titles] * len(configs)
    if isinstance(value_key, str):
        value_key = [value_key] * len(configs)
    if isinstance(value_label, str):
        value_label = [value_label] * len(configs)

    n_plots = len(configs)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, config in enumerate(configs):
        ax = axes_flat[i]
        v_key = value_key[i]
        v_label = value_label[i]
        
        points = get_points(config)
        points = filter_points(points, fixed_var, fixed_val)
        points = crop_points(points, x_key, xlim, y_key, ylim)
        points = [p for p in points if p[v_key] is not None]

        if not points:
            print(f'No data fits the chosen parameter constraints for config index {i} ({v_key})')
            ax.axis('off')
            continue

        values = np.array([p[v_key] for p in points])
        vmin, vmax = np.nanmin(values), np.nanmax(values)

        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.01, 1e-12)
            vmin, vmax = vmin - eps, vmax + eps

        xs, ys, grid = nearest_grid(points, x_key, y_key, v_key, xlim=xlim, ylim=ylim)

        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        
        scatter = ax.scatter([p[x_key] for p in points], [p[y_key] for p in points],
                             c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=65, edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titles[i] if i < len(titles) else titles[-1])

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(v_label, fontsize=plt.rcParams['axes.labelsize'])
        cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_diagram_int(configs, x_key, y_key, fixed_var, fixed_val, xlabel, ylabel, titles, filename=None, cmap='viridis', xlim=None, ylim=None):
    if isinstance(configs, dict):
        configs = [configs]
    if isinstance(titles, str):
        titles = [titles] * len(configs)

    n_plots = len(configs)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, config in enumerate(configs):
        ax = axes_flat[i]
        points = get_points(config)
        points = filter_points(points, fixed_var, fixed_val)
        points = crop_points(points, x_key, xlim, y_key, ylim)
        points = [p for p in points if p["chern"] is not None]

        if not points:
            print(f'No data fits the chosen parameter constraints for config index {i} (chern)')
            ax.axis('off')
            continue

        chern_vals = [p["chern"] for p in points]
        lo = int(round(min(chern_vals)))
        hi = int(round(max(chern_vals)))

        if lo == hi:
            lo -= 1
            hi += 1

        xs, ys, grid = nearest_grid(points, x_key, y_key, "chern", xlim=xlim, ylim=ylim)
        
        num_classes = max(1, hi - lo + 1)
        plot_cmap = plt.get_cmap(cmap, num_classes)
        bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
        norm = BoundaryNorm(bounds, plot_cmap.N)

        ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  aspect="auto", cmap=plot_cmap, norm=norm, zorder=1)

        scatter = ax.scatter([p[x_key] for p in points], [p[y_key] for p in points],
                             c=chern_vals, cmap=plot_cmap, norm=norm, s=65,
                             edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titles[i] if i < len(titles) else titles[-1])

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        ticks = list(range(lo, hi + 1))
        cbar = fig.colorbar(scatter, ax=ax, ticks=ticks, boundaries=bounds)
        cbar.set_label(r"$\mathrm{Chern\ number}$", fontsize=plt.rcParams['axes.labelsize'])
        cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_1d_cut(configs, x_key, fixed_vars, value_key, ylabel, xlabel, title, filename=None, colors=None, labels=None, xlim=None):
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

    # Handle ylabel and title mapping for each observable subplot
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

    for plot_idx, obs in enumerate(value_key):
        ax = axes_flat[plot_idx]
        lines_plotted = 0

        for i, config in enumerate(configs):
            points = get_points(config)
            
            for f_var, f_val in fixed_vars.items():
                points = filter_points(points, f_var, f_val)
                
            points = crop_points(points, x_key, xlim)
            points = [p for p in points if p[obs] is not None]

            if not points:
                print(f'No data fits the chosen parameter constraints for config index {i} and observable {obs}')
                continue

            points.sort(key=lambda p: p[x_key])

            x_vals = [p[x_key] for p in points]
            y_vals = [p[obs] for p in points]

            current_color = colors[i % len(colors)]
            current_label = labels[i] if i < len(labels) else None

            ax.plot(x_vals, y_vals, '-o', markersize=6, color=current_color, label=current_label)
            lines_plotted += 1

        if lines_plotted > 0:
            if plot_idx == 0: ax.set_title(title_dict.get(obs, str(obs)))
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


def plot_all(configs, u_range=None, delta_range=None, v_range=None):
    if not configs:
        return

    # --- 2D Phase Diagrams ---
    plot_diagram_int(configs, "u", "delta", "v", 0.0, r"$U/t$", r"$\Delta/t$",
                     r"$\mathrm{Chern\ number}:\ \Delta/t\ \mathrm{vs.}\ U/t,\ V/t=0$",
                     "chern_delta_u.png", xlim=u_range, ylim=delta_range)

    plot_diagram_int(configs, "u", "v", "delta", 0.0, r"$U/t$", r"$V/t$",
                     r"$\mathrm{Chern\ number}:\ V/t\ \mathrm{vs.}\ U/t,\ \Delta/t=0$",
                     "chern_v_u.png", xlim=u_range, ylim=v_range)

    # Added E_gs to automated loops alongside cdw, sdw, and gap
    for key, label in (("E_gs", "E_{gs}"), ("cdw", "CDW"), ("sdw", "SDW"), ("gap", "Gap")):
        val_label = rf"$\mathrm{{{label}}}$"
        
        plot_diagram(configs, "u", "delta", "v", 0.0, key, val_label, r"$U/t$", r"$\Delta/t$",
                   rf"$\mathrm{{{label}}}:\ \Delta/t\ \mathrm{{vs.}}\ U/t,\ V/t=0$",
                   f"{key}_delta_u.png", xlim=u_range, ylim=delta_range)

        plot_diagram(configs, "u", "v", "delta", 0.0, key, val_label, r"$U/t$", r"$V/t$",
                   rf"$\mathrm{{{label}}}:\ V/t\ \mathrm{{vs.}}\ U/t,\ \Delta/t=0$",
                   f"{key}_v_u.png", xlim=u_range, ylim=v_range)
                   
    # --- 1D Cuts (Examples) ---
    for key, label in (("E_gs", "E_{gs}"), ("cdw", "CDW"), ("sdw", "SDW"), ("gap", "Gap"), ("chern", "Chern number")):
        val_label = rf"$\mathrm{{{label}}}$" if key != "chern" else label
        
        # Example 1: Sweep U while Delta=0, V=0
        plot_1d_cut(configs, "u", {"delta": 0.0, "v": 0.0}, key, val_label, r"$U/t$", 
                    rf"{val_label}\ \mathrm{{vs.}}\ U/t\ (\Delta/t=0, V/t=0)$",
                    f"1d_{key}_vs_u.png", xlim=u_range)
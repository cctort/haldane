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


def get_auto_label(key, modifier=None):
    if key is None: return ""
    base_labels = {
        "u": r"$U/t$", "v": r"$V/t$", "delta": r"$\Delta/t$",
        "cdw": r"$\mathcal{S}_{\mathrm{CDW}}$", "sdw": r"$\mathcal{S}_{\mathrm{SDW}}$",
        "gap": r"$1/\epsilon_{\mathrm{gap}}$", "chern": r"$\mathcal{C}$",
        "resta": r"$P_{\mathrm{Resta}}$", "E_gs": r"$E_{\mathrm{gs}}/t$",
    }

    if key in base_labels: return base_labels[key]

    base = key[:-4] if len(key) > 4 else key
    inner = base_labels.get(base, rf"${base}$").strip("$")
    if modifier == 'avg' or key == 'chern':
        return rf'$\mathbb{{E}}[{inner}]$'
    elif modifier == 'log_avg':
        return rf'$\exp{{\mathbb{{E}}[\ln({inner})]}}$'
    elif modifier == 'std':
        return rf'$\sqrt{{Var[{inner}]}}$'
    elif modifier == 'rel_std':
        return rf'$\sqrt{{Var[{inner}]}}/\mathbb{{E}}[{inner}]$'

    return rf"${key}$"


def find_matching_file(target_config, data_dir):
    registry_path = Path(data_dir) / REGISTRY_FILE
    if not registry_path.exists(): return None

    with open(registry_path, 'r') as f:
        try: registry = json.load(f)
        except json.JSONDecodeError: return None

    is_match = lambda a, b: np.isclose(a, b, rtol=1e-5, atol=1e-15) if isinstance(a, (int, float)) else a == b
    for filename, config in registry.items():
        if all(k in config and is_match(config[k], v) for k, v in target_config.items()):
            path = Path(data_dir) / filename
            return path if path.exists() else None
    return None


def load_data(data_dir, config, fixed_vars=None, x_key=None, xlim=None, y_key=None, ylim=None):
    """Loads and filters point dictionaries from HDF5."""
    file_path = find_matching_file(config, data_dir)
    if not file_path: return []

    points = []
    with h5py.File(file_path, "r") as f:
        for grp in f.values():
            attrs_lower = {k.lower(): v for k, v in grp.attrs.items()}
            if all(k in attrs_lower for k in ("delta", "u", "v")):
                p = {k: float(v) if k in ("delta", "u", "v") else v for k, v in attrs_lower.items()}

                for k in grp.keys():
                    kl = k.lower()
                    if kl not in p:
                        val = grp[k][()]
                        p[kl] = val.item() if isinstance(val, np.ndarray) and val.ndim == 0 else val
                points.append(p)

    if fixed_vars:
        for k, v in fixed_vars.items():
            points = [p for p in points if k in p and np.isclose(p[k], v)]

    if xlim and x_key:
        points = [p for p in points if x_key in p and xlim[0] <= p[x_key] <= xlim[1]]
    if ylim and y_key:
        points = [p for p in points if y_key in p and ylim[0] <= p[y_key] <= ylim[1]]

    return points


def nearest_grid(points, x_key, y_key, value_key, n=400, xlim=None, ylim=None):
    x, y, vals = (np.array([p[k] for p in points]) for k in (x_key, y_key, value_key))
    x0, x1 = xlim or (x.min(), x.max())
    y0, y1 = ylim or (y.min(), y.max())

    xg, yg = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xg, yg)
    nearest = np.argmin((X[..., None] - x)**2 + (Y[..., None] - y)**2, axis=2)
    return xg, yg, vals[nearest].astype(float)


def _find_value(p, key):
    key = key.lower()
    candidates = [key, key + "_avg", key + "_mean"]
    if key == "chern":
        candidates.extend(["c", "chern_number", "chern_numbers", "chern_samples", "c_avg"])

    for c in candidates:
        if c in p: return p[c]

    for k in p:
        if k.startswith(key) and not k.endswith(("_std", "_err", "_var")):
            return p[k]
    return None


def _get_obs_val(p, key, modifier='avg'):
    key = key.lower()
    val = _find_value(p, key)
    if val is None: return None

    if isinstance(val, np.ndarray):
        if val.ndim == 0: val = val.item()
        elif val.ndim == 1 and len(val) == 1: val = val[0]

    if not isinstance(val, (list, tuple, np.ndarray)) or np.array(val).ndim == 0:
        try: fval = float(val)
        except (ValueError, TypeError): return np.nan

        if modifier in ('std', 'rel_std'): return 0.0
        if key == 'gap': return 1.0 / fval if not np.isclose(fval, 0) else np.nan
        return fval

    val_arr = np.array(val, dtype=float).ravel()
    if key == 'gap':
        val_arr = np.array([1.0 / v if not np.isclose(v, 0) else np.nan for v in val_arr])

    if np.all(np.isnan(val_arr)): return np.nan
    avg = np.nanmean(val_arr)

    if modifier == 'avg' or key == 'chern': return avg
    elif modifier == 'log_avg':
        valid_val = val_arr[~np.isnan(val_arr)]
        return np.exp(np.nanmean(np.log(valid_val))) if np.all(valid_val > 0) else avg
    elif modifier == 'std': return np.nanstd(val_arr)
    elif modifier == 'rel_std': return np.nanstd(val_arr) / avg if avg and not np.isclose(avg, 0) else np.nan
    return avg


def _empty_panel(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_diagram(datasets, x_key, y_key, value_key, titles="", filename=None, cmap="hot", xlim=None, ylim=None, modifier='avg'):
    if datasets and isinstance(datasets[0], dict):
        datasets = [datasets]

    value_key = [value_key] if isinstance(value_key, str) else list(value_key)
    n_datasets, n_obs = len(datasets), len(value_key)

    modifier = [modifier] * n_obs if isinstance(modifier, str) or modifier is None else list(modifier)
    titles = [titles] * n_datasets if isinstance(titles, str) else titles

    x_lbl, y_lbl = get_auto_label(x_key), get_auto_label(y_key)
    val_lbls = [get_auto_label(k, m) for k, m in zip(value_key, modifier)]

    n_rows, n_cols = (1, n_obs) if n_datasets == 1 else (n_obs, n_datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False, layout='constrained')

    plot_data, obs_vals = {}, {i: [] for i in range(n_obs)}
    for d_idx, pts in enumerate(datasets):
        for o_idx, (v_key, mod) in enumerate(zip(value_key, modifier)):
            valid_pts = []
            for p in pts:
                val = _get_obs_val(p, v_key, mod)
                if val is not None and not np.isnan(val) and not (v_key == "gap" and val <= 0):
                    valid_pts.append({**p, v_key: val})
                    obs_vals[o_idx].append(val)
            plot_data[(d_idx, o_idx)] = valid_pts

    plot_kws = {}
    for o_idx, v_key in enumerate(value_key):
        vals = obs_vals[o_idx]
        if not vals:
            plot_kws[o_idx] = None
            continue

        vmin, vmax = min(vals), max(vals)
        kws = {'cmap': cmap}
        is_integer = np.allclose(vals, np.round(vals), atol=1e-5)

        if is_integer:
            lo, hi = int(round(vmin)), int(round(vmax))
            if lo == hi: lo, hi = lo - 1, hi + 1
            bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
            kws['cmap'] = plt.get_cmap("viridis" if cmap == "hot" else cmap, hi - lo + 1)
            kws.update({'norm': BoundaryNorm(bounds, kws['cmap'].N), 'cbar_ticks': list(range(lo, hi + 1)), 'cbar_boundaries': bounds})
        elif v_key == "gap":
            vmin = max(vmin, 1e-12) if vmin <= 0 else vmin
            kws['norm'] = LogNorm(vmin=vmin, vmax=max(vmax, vmin + 1e-12))
        else:
            eps = max(abs(vmin) * 0.01, 1e-12) if np.isclose(vmin, vmax) else 0
            kws.update({'vmin': vmin - eps, 'vmax': vmax + eps})
        plot_kws[o_idx] = kws

    for o_idx, v_key in enumerate(value_key):
        kws = plot_kws.get(o_idx)

        if not kws:
            for d_idx in range(n_datasets):
                ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, d_idx]
                _empty_panel(ax, f"no data for '{v_key}'")
            continue

        draw_kws = {k: v for k, v in kws.items() if not k.startswith('cbar_')}
        cb_kws = {k[5:]: v for k, v in kws.items() if k.startswith('cbar_')}
        last_sc = None

        for d_idx in range(n_datasets):
            ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, d_idx]
            obs_pts = plot_data[(d_idx, o_idx)]

            if not obs_pts:
                _empty_panel(ax, f"no data for '{v_key}'")
                continue

            vals = [p[v_key] for p in obs_pts]
            xs, ys, grid = nearest_grid(obs_pts, x_key, y_key, v_key, xlim=xlim, ylim=ylim)
            sz = 65 / (len(obs_pts) / 441.0)

            ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto", zorder=1, **draw_kws)
            sc = ax.scatter([p[x_key] for p in obs_pts], [p[y_key] for p in obs_pts], c=vals, s=sz, edgecolors="black", linewidths=0.8, zorder=4, **draw_kws)
            last_sc = sc

            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)

            if n_datasets == 1 or o_idx == 0:
                ax.set_title(titles[d_idx] if d_idx < len(titles) else titles[-1], pad=plt.rcParams["axes.titlesize"]//2)

        if last_sc is not None:
            cb_ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, :].tolist()
            cbar = fig.colorbar(last_sc, ax=cb_ax, pad=0.015, **cb_kws)
            cbar.set_label(val_lbls[o_idx], fontsize=plt.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])

    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes


def plot_1d_cut(datasets, x_key, value_key, title="", filename=None, colors=None, labels=None, xlim=None, relative_to=None, relative_std=False, modifier='avg'):
    """Plots 1D cuts. 'datasets' is either a list of point dicts or a list of such lists."""
    if datasets and isinstance(datasets[0], dict):
        datasets = [datasets]

    value_key = [value_key] if isinstance(value_key, str) else list(value_key)
    n_datasets, n_plots = len(datasets), len(value_key)

    if relative_std:
        modifier = ['rel_std'] * n_plots
    else:
        modifier = [modifier] * n_plots if isinstance(modifier, str) or modifier is None else list(modifier)

    colors = colors if isinstance(colors, list) else [colors or COLOR_LIST[0]] * n_datasets
    labels = labels if isinstance(labels, list) else [labels] * n_datasets

    title_dict = title if isinstance(title, dict) else {k: (title[i] if isinstance(title, list) else title) for i, k in enumerate(value_key)}
    rows, cols = min(3, n_plots), math.ceil(n_plots / min(3, n_plots))

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 1 + 2 * rows), sharex=True, layout='constrained')
    axes_flat = np.atleast_1d(axes).flatten()

    cache = [sorted(pts, key=lambda p: p[x_key]) for pts in datasets]

    indices = list(range(n_datasets))
    if relative_to is not None:
        indices.remove(relative_to)
        indices.insert(0, relative_to)

    for p_idx, (obs, mod) in enumerate(zip(value_key, modifier)):
        ax, lines_plotted, ref_vals = axes_flat[p_idx], 0, {}

        for loop_idx, d_idx in enumerate(indices):
            pts = cache[d_idx]

            valid = []
            for p in pts:
                v = _get_obs_val(p, obs, mod)
                err = _get_obs_val(p, obs + "_std", 'avg') if (obs + "_std") in p else _get_obs_val(p, obs, 'std')

                if v is not None and np.isfinite(v):
                    valid.append((p[x_key], v, err))

            if relative_to is not None and loop_idx == 0:
                ref_vals = {x: y for x, y, _ in valid}
                continue

            x_vals, y_vals, y_errs = [], [], []
            for x, y, err in valid:
                if relative_to is not None:
                    if x not in ref_vals: continue
                    y -= ref_vals[x]
                x_vals.append(x)
                y_vals.append(y)
                y_errs.append(err if err is not None and not np.isnan(err) else 0.0)

            if not x_vals: continue

            c, l = colors[d_idx % len(colors)], labels[d_idx]
            ax.plot(x_vals, y_vals, '-o', markersize=6, color=c, label=l)
            if any(y_errs):
                y_v, y_e = np.array(y_vals), np.array(y_errs)
                ax.fill_between(x_vals, y_v - y_e, y_v + y_e, color=c, alpha=0.25)

            lines_plotted += 1

        if lines_plotted > 0:
            if p_idx == 0: ax.set_title(title_dict.get(obs, str(obs)), pad=plt.rcParams["axes.titlesize"]//2)
            if p_idx == n_plots - 1: ax.set_xlabel(get_auto_label(x_key))
            ax.set_ylabel(get_auto_label(obs, modifier=mod))
            if xlim: ax.set_xlim(xlim)
            ax.grid(True, linestyle='--', alpha=0.6)
            if any(labels) and p_idx == 0: ax.legend()
        else:
            _empty_panel(ax, f"no data for '{obs}'")

    for ax in axes_flat[n_plots:]: ax.set_visible(False)

    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes
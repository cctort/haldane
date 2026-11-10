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

from ed.config import Config
from ed.lattice import Lattice, get_latt_sizes, get_latt_vecs

REGISTRY_FILE = "configs.json"
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

COLOR_LIST = sns.color_palette('colorblind') + sns.color_palette("Set2") + sns.color_palette("Set3")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

BASE_LABELS = {
    "u": r"$U/t$", "v": r"$V/t$", "delta": r"$\Delta/t$",
    "e_gs": r"$E_{\mathrm{gs}}/t$",
    "gap": r"$\epsilon_{\mathrm{gap}}/t$",
    "inv_gap": r"$(\epsilon_{\mathrm{gap}}/t)^{-1}$",
    "chern": r"$\mathcal{C}$",
    "ss_z_q0": r"$\mathcal{S}^z(0,0)$",
    "ss_z_qpi": r"$\mathcal{S}^z(\pi,\pi)$",
    "ss_x_q0": r"$\mathcal{S}^x(0,0)$",
    "ss_x_qpi": r"$\mathcal{S}^x(\pi,\pi)$",
    "sc_q0": r"$\mathcal{N}(0,0)$",
    "sc_qpi": r"$\mathcal{N}(\pi,\pi)$",
    "chi_z_sg": r"$\chi_{\mathrm{SG}}^z$",
    "chi_z_sg_off": r"$\chi_{\mathrm{SG,off}}^z$",
    "chi_x_sg": r"$\chi_{\mathrm{SG}}^x$",
    "chi_x_sg_off": r"$\chi_{\mathrm{SG,off}}^x$",
    "chi_cg": r"$\chi_{\mathrm{CG}}$",
    "chi_cg_off": r"$\chi_{\mathrm{CG,off}}$",
}

DERIVED = {
    "inv_gap": ("gap", lambda a: np.where(np.isclose(a, 0), np.nan, 1.0 / np.where(np.isclose(a, 0), 1.0, a))),
}

def get_auto_label(key, modifier=None):
    if key is None: return ""
    key = key.lower()
    inner = BASE_LABELS.get(key, rf"${key}$").strip("$")

    if modifier in (None, 'avg') or key == 'chern':
        return BASE_LABELS.get(key, rf"${key}$")
    elif modifier == 'log_avg':
        return rf'$\exp{{\mathbb{{E}}[\ln({inner})]}}$'
    elif modifier == 'std':
        return rf'$\sqrt{{Var[{inner}]}}$'
    elif modifier == 'rel_std':
        return rf'$\sqrt{{Var[{inner}]}}/\mathbb{{E}}[{inner}]$'

    return BASE_LABELS.get(key, rf"${key}$")


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


def translation_perms(lat, tol=1e-4):
    """Site permutations of all lattice translations that map the cluster onto itself.

    Returns an array (n_translations, N), identity included; perms[t][i] is the site that site i
    is sent to by translation t. Built numerically from the primitive vectors (a1, a2) and the
    periodic supercell (l1, l2) used by Lattice. A translation is kept only if every site lands on
    a site of the same sublattice, so a cluster whose supercell is not commensurate with the
    lattice just gets fewer translations (n_translations < N / 2 triggers a warning).
    """
    l1, l2 = get_latt_sizes(lat.cluster, lat.n_max)
    a1, a2, _, _ = get_latt_vecs(lat.cell)
    M_inv = np.linalg.inv(np.array([l1, l2]).T)

    def wrap(pos):
        c = pos @ M_inv.T
        return c - np.floor(c + 1e-5)              # same [-1e-5, 1-1e-5) convention as Lattice

    frac = wrap(lat.positions)
    A_inv = np.linalg.inv(np.array([a1, a2]).T)
    R = int(np.ceil(np.abs(A_inv @ np.array([l1, l2]).T).max())) + 1

    perms = {}
    for n1 in range(-R, R + 1):
        for n2 in range(-R, R + 1):
            d = wrap(lat.positions + n1 * a1 + n2 * a2)[:, None, :] - frac[None, :, :]
            d -= np.round(d)                       # periodic distance in fractional coordinates
            match = np.all(np.abs(d) < tol, axis=-1)
            if not np.all(match.sum(axis=1) == 1):
                continue                           # translation does not map the cluster onto itself
            p = match.argmax(axis=1)
            if len(set(p)) == lat.n_sites and np.all(lat.signs[p] == lat.signs):
                perms[p.tobytes()] = p
    out = np.array(list(perms.values()))
    n_cells = lat.n_sites // len(get_latt_vecs(lat.cell)[2])
    if len(out) < n_cells:
        print(f"warning: only {len(out)} of {n_cells} translations are symmetries of this cluster; "
              "chi_*_sg_off only removes order invariant under that subgroup")
    return out


def one_point_means(C, total):
    """<O_i> from the sum rule sum_j <O_i O_j> = total * <O_i> (fixed particle number or S^z_tot).
    C: raw correlations (..., N, N). Returns zeros when total == 0, where <O_i> = 0 by symmetry."""
    if not total:
        return np.zeros(C.shape[:-1])
    return C.sum(axis=-1) / total


def glass_chi(corr, perms, total=0.0):
    """Connected squared correlations per sample, C_c = <O_i O_j> - <O_i><O_j>.

    <O_i> comes from the sum rule with `total` (N_el for charge, S^z_tot for S^z); total = 0
    means nothing is subtracted. P = average over all lattice translations, the orthogonal
    projector onto the translation-invariant part (the q = q' block).
    Returns (chi, chi_off):
      chi     = (1/N) ||C_c||_F^2            full
      chi_off = (1/N) ||C_c - P(C_c)||_F^2   non-translation-invariant part only
    corr: (..., N, N), stored as <O_i O_j> / N.
    """
    N = corr.shape[-1]
    C = corr * N
    m = one_point_means(C, total)
    C = C - m[..., :, None] * m[..., None, :]
    P = sum(C[..., p[:, None], p[None, :]] for p in perms) / len(perms)
    return np.sum(C ** 2, axis=(-2, -1)) / N, np.sum((C - P) ** 2, axis=(-2, -1)) / N


def s_q0(corr):
    """Sum over all spatial correlation elements (q=0 structure factor)."""
    return np.sum(corr, axis=(-2, -1))


def stagger(corr_mat, signs):
    """Multiply correlation matrix by sublattice sign factors for q=pi."""
    stagg_mat = signs[:, None] * signs[None, :]
    return stagg_mat * corr_mat


def s_qpi(corr, signs):
    """Staggered sum over spatial correlation elements (q=pi structure factor)."""
    return np.sum(stagger(corr, signs), axis=(-2, -1))


def structure_factors(p, perms, signs, filling=None, sz=0.0):
    """Per-sample observables (arrays over disorder samples); modifiers act on them in get_obs_val."""
    if "corr_sz" in p:
        C_sz = np.asarray(p["corr_sz"], dtype=float)
        p["ss_z_q0"] = s_q0(C_sz)
        p["ss_z_qpi"] = s_qpi(C_sz, signs)
        if perms is not None:
            p["chi_z_sg"], p["chi_z_sg_off"] = glass_chi(C_sz, perms, sz)

    if "corr_sx" in p:
        C_sx = np.asarray(p["corr_sx"], dtype=float)
        p["ss_x_q0"] = s_q0(C_sx)
        p["ss_x_qpi"] = s_qpi(C_sx, signs)
        if perms is not None:   # <S^x_i> = 0 in an S^z_tot eigenstate, even for S^z_tot != 0
            p["chi_x_sg"], p["chi_x_sg_off"] = glass_chi(C_sx, perms)

    if "corr_ch" in p:
        C_raw = np.asarray(p["corr_ch"], dtype=float)
        p["sc_q0"] = s_q0(C_raw)
        p["sc_qpi"] = s_qpi(C_raw, signs)
        if perms is not None and filling is not None:
            p["chi_cg"], p["chi_cg_off"] = glass_chi(C_raw, perms, 2 * filling * C_raw.shape[-1])
        elif perms is not None:
            print("warning: no FILLING attribute, chi_cg unavailable (needs N_el for the connected part)")


def load_data(data_dir, config, fixed_vars=None, x_key=None, xlim=None, y_key=None, ylim=None):
    file_path = find_matching_file(config, data_dir)
    if not file_path: return []

    points = []
    with h5py.File(file_path, "r") as f:
        signs = None
        perms = None
        filling = float(f.attrs["FILLING"]) if "FILLING" in f.attrs else None
        sz = float(f.attrs.get("SZ", f.attrs.get("sz", 0.0)))
        if all(k in f.attrs for k in ("CELL", "CLUSTER", "N_MAX")):
            try:
                cfg_temp = Config(
                    cell=str(f.attrs["CELL"]),
                    cluster=str(f.attrs["CLUSTER"]),
                    n_max=list(f.attrs["N_MAX"])
                )
                lat = Lattice(cfg_temp)
                signs = np.asarray(lat.signs, dtype=float)
            except Exception as e:
                print(f"warning: could not build lattice, staggered (q=pi) observables unavailable ({e})")
            else:
                try:
                    perms = translation_perms(lat)
                except Exception as e:
                    print(f"warning: no translations, chi_*_sg_off / chi_cg_off unavailable ({e})")

        for grp in f.values():
            attrs_lower = {k.lower(): v for k, v in grp.attrs.items()}
            if all(k in attrs_lower for k in ("delta", "u", "v")):
                p = {k: float(v) if k in ("delta", "u", "v") else v for k, v in attrs_lower.items()}

                for k in grp.keys():
                    kl = k.lower()
                    if kl not in p:
                        val = grp[k][()]
                        p[kl] = val.item() if isinstance(val, np.ndarray) and val.ndim == 0 else val
                if signs is not None:
                    structure_factors(p, perms, signs, filling, sz)
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


def find_value(p, key):
    key = key.lower()
    candidates = [key, key + "_avg", key + "_mean"]
    if key == "chern":
        candidates.extend(["c", "chern_number", "chern_numbers", "chern_samples", "c_avg"])

    for c in candidates:
        if c in p: return p[c]
    return None


def get_obs_val(p, key, modifier='avg'):
    key = key.lower()

    transform = None
    if key in DERIVED:
        raw_key, transform = DERIVED[key]
        val = find_value(p, raw_key)
    else:
        val = find_value(p, key)
    if val is None: return None

    val_arr = np.atleast_1d(np.array(val, dtype=float)).ravel()
    if transform is not None:
        val_arr = transform(val_arr)

    if np.all(np.isnan(val_arr)): return np.nan
    if isinstance(modifier, (int, np.integer)):    # n-th disorder realization (0-based), no averaging
        return val_arr[modifier] if -len(val_arr) <= modifier < len(val_arr) else np.nan
    if modifier == 'mode':                         # most frequent value, for integer observables (ties: smallest)
        vals, counts = np.unique(val_arr[~np.isnan(val_arr)], return_counts=True)
        return vals[counts.argmax()]
    avg = np.nanmean(val_arr)

    if modifier == 'avg' or key == 'chern': return avg
    elif modifier == 'log_avg':
        valid_val = val_arr[~np.isnan(val_arr)]
        return np.exp(np.mean(np.log(valid_val))) if np.all(valid_val > 0) else avg
    elif modifier == 'std': return np.nanstd(val_arr)
    elif modifier == 'rel_std': return np.nanstd(val_arr) / avg if avg and not np.isclose(avg, 0) else np.nan
    return avg


def as_list(modifier, n):
    """One modifier (None, str or realization index) is repeated for all n observables."""
    return [modifier] * n if modifier is None or isinstance(modifier, (str, int, np.integer)) else list(modifier)


def diagnose(points, keys=("chi_z_sg", "chi_z_sg_off", "chi_x_sg", "chi_x_sg_off", "chi_cg", "chi_cg_off"), modifiers=("avg", "std")):
    """Print, per key, why a plot might be empty: missing data, no sample axis, NaN or zero values."""
    print(f"{len(points)} points")
    for key in keys:
        corr_key = {"z": "corr_sz", "x": "corr_sx", "c": "corr_ch"}[key[4]]   # chi_z_*, chi_x_*, chi_cg*
        has_key = sum(key in p for p in points)
        shapes = sorted({np.shape(p[corr_key]) for p in points if corr_key in p})[:3]
        print(f"[{key}] in {has_key}/{len(points)} points | {corr_key} shapes (first 3): {shapes}")
        for mod in modifiers:
            vals = np.array([np.nan if (v := get_obs_val(p, key, mod)) is None else v for p in points], dtype=float)
            fin = vals[np.isfinite(vals)]
            msg = f"    {mod:>4}: finite {fin.size}/{vals.size}"
            if fin.size:
                msg += f" | min {fin.min():.3g} max {fin.max():.3g} | exactly zero: {np.sum(fin == 0)}"
            print(msg)


def empty_panel(ax, msg="no data"):
    ax.text(0.5, 0.5, msg, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_diagram(datasets, x_key, y_key, value_key, labels=None, title="", filename=None, cmap="hot", xlim=None, ylim=None, modifier='avg', log_cbar=False):
    if datasets and isinstance(datasets[0], dict):
        datasets = [datasets]

    value_key = [value_key] if isinstance(value_key, str) else list(value_key)
    n_datasets, n_obs = len(datasets), len(value_key)

    modifier = as_list(modifier, n_obs)
    labels = [labels] * n_datasets if isinstance(labels, str) else labels
    log_cbar = [bool(log_cbar)] * n_obs if isinstance(log_cbar, (bool, np.bool_)) else [bool(b) for b in log_cbar]

    x_lbl, y_lbl = get_auto_label(x_key), get_auto_label(y_key)
    val_lbls = [get_auto_label(k, m) for k, m in zip(value_key, modifier)]

    n_rows, n_cols = (1, n_obs) if n_datasets == 1 else (n_obs, n_datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.3 * n_cols, 4.6 * n_rows), squeeze=False, layout='constrained')

    plot_data, obs_vals = {}, {i: [] for i in range(n_obs)}
    for d_idx, pts in enumerate(datasets):
        for o_idx, (v_key, mod) in enumerate(zip(value_key, modifier)):
            valid_pts = []
            for p in pts:
                val = get_obs_val(p, v_key, mod)
                if val is not None and not np.isnan(val) and not (log_cbar[o_idx] and val <= 0):
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
        is_integer = np.allclose(vals, np.round(vals), atol=1e-5) and not np.allclose(0., np.round(vals), atol=1e-6)

        if log_cbar[o_idx]:
            kws['norm'] = LogNorm(vmin=vmin, vmax=max(vmax, vmin * (1 + 1e-9)))
        elif is_integer:
            lo, hi = int(round(vmin)), int(round(vmax))
            if lo == hi: lo, hi = lo - 1, hi + 1
            bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
            kws['cmap'] = plt.get_cmap("viridis" if cmap == "hot" else cmap, hi - lo + 1)
            kws.update({'norm': BoundaryNorm(bounds, kws['cmap'].N), 'cbar_ticks': list(range(lo, hi + 1)), 'cbar_boundaries': bounds})
        else:
            eps = max(abs(vmin) * 0.01, 1e-12) if np.isclose(vmin, vmax) else 0
            kws.update({'vmin': vmin - eps, 'vmax': vmax + eps})
        plot_kws[o_idx] = kws

    for o_idx, v_key in enumerate(value_key):
        kws = plot_kws.get(o_idx)

        if not kws:
            for d_idx in range(n_datasets):
                ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, d_idx]
                empty_panel(ax, f"no data for '{v_key}'")
            continue

        draw_kws = {k: v for k, v in kws.items() if not k.startswith('cbar_')}
        cb_kws = {k[5:]: v for k, v in kws.items() if k.startswith('cbar_')}
        last_sc = None

        for d_idx in range(n_datasets):
            ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, d_idx]
            obs_pts = plot_data[(d_idx, o_idx)]

            if not obs_pts:
                empty_panel(ax, f"no data for '{v_key}'")
                continue

            vals = [p[v_key] for p in obs_pts]
            xs, ys, grid = nearest_grid(obs_pts, x_key, y_key, v_key, xlim=xlim, ylim=ylim)
            sz = 45 / (len(obs_pts) / 441.0)

            ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto", zorder=1, **draw_kws)
            sc = ax.scatter([p[x_key] for p in obs_pts], [p[y_key] for p in obs_pts], c=vals, s=sz, edgecolors="black", linewidths=0.8, zorder=4, **draw_kws)
            last_sc = sc

            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)

            if o_idx == 0:
                ax.set_title(labels[d_idx], pad=plt.rcParams["axes.titlesize"] // 2)

        if last_sc is not None:
            cb_ax = axes[0, o_idx] if n_datasets == 1 else axes[o_idx, :].tolist()
            cbar = fig.colorbar(last_sc, ax=cb_ax, pad=0.015, **cb_kws)
            cbar.set_label(val_lbls[o_idx], fontsize=plt.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize'])
    if title:
        fig.get_layout_engine().set(h_pad=0.1)
        fig.suptitle(title, fontsize=plt.rcParams["axes.titlesize"] * 1.2)
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes


def plot_1d_cut(datasets, x_key, value_key, title="", filename=None, colors=None, labels=None, xlim=None, relative_to=None, relative_std=False, modifier='avg'):
    if datasets and isinstance(datasets[0], dict):
        datasets = [datasets]

    value_key = [value_key] if isinstance(value_key, str) else list(value_key)
    n_datasets, n_plots = len(datasets), len(value_key)

    if relative_std:
        modifier = ['rel_std'] * n_plots
    else:
        modifier = as_list(modifier, n_plots)

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
                v = get_obs_val(p, obs, mod)
                if isinstance(mod, (int, np.integer)) or mod == 'mode':
                    err = None                     # spread over realizations means nothing for one realization / the mode
                else:
                    err = get_obs_val(p, obs + "_std", 'avg') if (obs + "_std") in p else get_obs_val(p, obs, 'std')

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
            empty_panel(ax, f"no data for '{obs}'")

    for ax in axes_flat[n_plots:]: ax.set_visible(False)

    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes

def get_obs_samples(p, key):
    """Return the realization values of an observable."""
    key = key.lower()
    transform = None
    if key in DERIVED:
        raw_key, transform = DERIVED[key]
        val = find_value(p, raw_key)
    else:
        val = find_value(p, key)

    if val is None:
        return np.array([], dtype=float)

    values = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
    if transform is not None:
        values = transform(values)
    return values[np.isfinite(values)]


def _normalise_param_points(param_points, x_key="u", y_key="v"):
    """Convert selected point(s) to parameter dictionaries."""
    if isinstance(param_points, dict):
        return [dict(param_points)]
    if isinstance(param_points, np.ndarray):
        param_points = param_points.tolist()
    if not isinstance(param_points, (list, tuple)):
        raise TypeError("param_points must be a dict, an (x, y) pair, or a list of points")
    if len(param_points) == 0:
        raise ValueError("param_points cannot be empty")

    if len(param_points) == 2 and all(
        np.isscalar(item) and not isinstance(item, dict) for item in param_points
    ):
        return [{x_key: param_points[0], y_key: param_points[1]}]

    selectors = []
    for point in param_points:
        if isinstance(point, dict):
            selectors.append(dict(point))
        else:
            if isinstance(point, np.ndarray):
                point = point.tolist()
            if isinstance(point, (list, tuple)) and len(point) == 2:
                selectors.append({x_key: point[0], y_key: point[1]})
            else:
                raise ValueError("Each point must be a dict or an (x, y) pair")
    return selectors


def _matches_param_selector(point, selector, rtol=1e-5, atol=1e-12):
    for key, target in selector.items():
        if key not in point:
            return False
        try:
            matches = np.isclose(float(point[key]), float(target), rtol=rtol, atol=atol)
        except (TypeError, ValueError):
            matches = point[key] == target
        if not matches:
            return False
    return True


def _format_param_selector(selector):
    def fmt(value):
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)
    return ", ".join(f"{key}={fmt(value)}" for key, value in selector.items())


def plot_histograms(datasets, value_key, param_points, labels=None, title="", bins=20, density=True, filename=None):
    """Plot realization-level histograms at selected parameter points."""
    if datasets and isinstance(datasets[0], dict):
        datasets = [datasets]
    if not datasets:
        raise ValueError("No datasets supplied")

    value_keys = [value_key] if isinstance(value_key, str) else list(value_key)
    if not value_keys:
        raise ValueError("value_key cannot be empty")
    selectors = _normalise_param_points(param_points)
    n_datasets, n_points, n_obs = len(datasets), len(selectors), len(value_keys)

    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(n_datasets)]
    elif isinstance(labels, str):
        labels = [labels] * n_datasets
    else:
        labels = list(labels)
    if len(labels) != n_datasets:
        raise ValueError(f"Expected {n_datasets} labels, got {len(labels)}")

    point_labels = [_format_param_selector(selector) for selector in selectors]
    matched_points = {}
    for d_idx, pts in enumerate(datasets):
        for p_idx, selector in enumerate(selectors):
            axes = [key for key in ("u", "v") if key in selector]
            fixed = {key: val for key, val in selector.items() if key not in axes}
            candidates = [p for p in pts if _matches_param_selector(p, fixed)]
            if axes:
                candidates = [p for p in candidates if all(key in p and np.isfinite(float(p[key])) for key in axes)]
                if candidates:
                    target = np.array([float(selector[key]) for key in axes])
                    candidates = [min(candidates, key=lambda p: np.sum((np.array([float(p[key]) for key in axes]) - target) ** 2))]
            else:
                candidates = [p for p in pts if _matches_param_selector(p, selector)]
            matched_points[(d_idx, p_idx)] = candidates[0] if candidates else None

    fig, axes = plt.subplots(n_obs, n_datasets, figsize=(6.3 * n_datasets, 4.6 * n_obs), squeeze=False, layout="constrained")

    for o_idx, obs in enumerate(value_keys):
        all_values = [
            get_obs_samples(matched_points[(d_idx, p_idx)], obs)
            for d_idx in range(n_datasets)
            for p_idx in range(n_points)
            if matched_points[(d_idx, p_idx)] is not None
        ]
        all_values = [values for values in all_values if values.size]
        bin_edges = None
        if all_values:
            pooled_values = np.concatenate(all_values)
            bin_edges = np.histogram_bin_edges(pooled_values, bins=bins) if isinstance(bins, (int, np.integer, str)) else bins

        for d_idx in range(n_datasets):
            ax = axes[o_idx, d_idx]
            has_data = False
            for p_idx in range(n_points):
                point = matched_points[(d_idx, p_idx)]
                values = get_obs_samples(point, obs) if point is not None else np.array([])
                if not values.size:
                    continue
                ax.hist(values, bins=bin_edges, density=density, histtype="stepfilled", alpha=0.3,
                        label=point_labels[p_idx], color=COLOR_LIST[p_idx % len(COLOR_LIST)],
                        edgecolor="black", linewidth=0.8)
                has_data = True

            if not has_data:
                empty_panel(ax, f"no realization data for '{obs}'")
            else:
                ax.set_xlabel(get_auto_label(obs))
                ax.set_ylabel("Probability density" if density else "Counts")
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                ax.legend(fontsize=max(8, plt.rcParams["legend.fontsize"] - 2))
            if o_idx == 0:
                ax.set_title(labels[d_idx], pad=plt.rcParams["axes.titlesize"] // 2)

    if title:
        fig.get_layout_engine().set(h_pad=0.1)
        fig.suptitle(title, fontsize=plt.rcParams["axes.titlesize"] * 1.2)
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes

def plot_flux_grid(data):
    """Berry curvature (plaquette flux) over the twist-angle BZ.
    data: dict (observables.flux_data) or the NpzFile loaded from chern_grid.npz.
    curvature[ix, iy] is the flux through the plaquette whose lower-left corner is (phi_x[ix], phi_y[iy])."""
    px, py = data['phi_x'] / np.pi, data['phi_y'] / np.pi
    # periodic closure: the last grid line is followed by the first one shifted by 2 (units of pi)
    px_ext = np.append(px, px[0] + 2)
    py_ext = np.append(py, py[0] + 2)
    curv = data['curvature']
    lim = max(np.abs(curv).max(), 1e-3)

    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    m = ax.pcolormesh(px_ext, py_ext, curv.T, cmap='RdBu_r', vmin=-lim, vmax=lim)
    X, Y = np.meshgrid(px, py, indexing='ij')
    ax.scatter(X, Y, s=8, c='k')
    fig.colorbar(m, ax=ax, label='Berry curvature (plaquette flux, rad)')
    ax.set_title(f'grid {len(px)}x{len(py)}, ' + r'$\mathcal{C}$' + f' = {float(data["chern"]):.4f}')

    # point where the gap is measured
    gx, gy = data['gap_phi'] / np.pi
    if np.isfinite(gx):
        ax.plot(gx, gy, '*', color='red', ms=16, mec='k', zorder=4, label='gap measured')
        ax.legend(loc='upper right')
    ax.set_xlim(px_ext[0], px_ext[-1])
    ax.set_ylim(py_ext[0], py_ext[-1])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\phi_x/\pi$')
    ax.set_ylabel(r'$\phi_y/\pi$')
    return fig
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ed.lattice import Lattice, POSITIONS, SUB_SIGN


IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def plot_lattice():
    lattice = Lattice()

    a = 1.0
    sqrt3 = np.sqrt(3)

    # Real-space supercell
    L1 = np.array([6 * a, 0.0])
    L2 = np.array([1.5 * a, 1.5 * sqrt3 * a])

    # Primitive reciprocal lattice vectors
    b1 = np.array([2 * np.pi / (3 * a), 2 * np.pi / (sqrt3 * a)])
    b2 = np.array([2 * np.pi / (3 * a), -2 * np.pi / (sqrt3 * a)])

    # Reciprocal supercell vectors
    G1 = np.array([np.pi / (3 * a), -np.pi / (3 * sqrt3 * a)])
    G2 = np.array([0.0, 4 * np.pi / (3 * sqrt3 * a)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------
    # Real space
    # ------------------------------------------------------------
    ax = axes[0]

    cell = np.array([[0, 0], L1, L1 + L2, L2, [0, 0]])
    ax.plot(cell[:, 0], cell[:, 1], "--", color="black", lw=1.2)

    ax.scatter(POSITIONS[:, 0], POSITIONS[:, 1], c=SUB_SIGN, cmap="bwr", s=70, edgecolors="black", zorder=3)

    for i, j in lattice.nn_bonds:
        if lattice.nn_shift[(i, j)] == (0, 0):
            r1, r2 = POSITIONS[i], POSITIONS[j]
            ax.plot([r1[0], r2[0]], [r1[1], r2[1]], "--", color="black", lw=1.2)

    ax.set_xlabel(r"$x/a$")
    ax.set_ylabel(r"$y/a$")
    ax.set_title("Real space")
    ax.set_aspect("equal")

    # ------------------------------------------------------------
    # Reciprocal space
    # ------------------------------------------------------------
    ax = axes[1]

    # Reciprocal supercell boundary
    cell = np.array([[0, 0], G1, G1 + G2, G2, [0, 0]])
    ax.plot(cell[:, 0], cell[:, 1], "--", color="black", lw=1.2)

    # Six allowed momenta of the 12-site periodic cluster
    k_points = []
    for n in range(6):
        k = n * (b1 + 2 * b2) / 6
        k_points.append(k)

    k_points = np.array(k_points)
    ax.scatter(k_points[:, 0], k_points[:, 1], s=50, color="black", zorder=3)

    # Primitive-cell Gamma, K and M
    Gamma = np.array([0.0, 0.0])
    K = (2 * b1 + b2) / 3
    M = (b1 + b2) / 2

    ax.scatter([Gamma[0], K[0], M[0]], [Gamma[1], K[1], M[1]], s=100, facecolors="white", edgecolors="black", zorder=5)

    ax.text(Gamma[0], Gamma[1], r"$\Gamma$", fontsize=14, ha="right", va="top")
    ax.text(K[0], K[1], r"$K$", fontsize=14, ha="left", va="bottom")
    ax.text(M[0], M[1], r"$M$", fontsize=14, ha="left", va="bottom")

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("Reciprocal space")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "lattice.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_lattice()
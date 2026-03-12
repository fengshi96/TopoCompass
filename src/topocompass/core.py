from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Minimal self-contained model container
# ============================================================

@dataclass
class SpinExchangeModel:
    exchanges: Dict[str, Any]
    magnetic_field_xyz: np.ndarray | tuple[float, float, float]
    symmetry: str = "C3i"


# ============================================================
# Basic utilities
# ============================================================

def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    nrm = float(np.linalg.norm(vec))
    if nrm < eps:
        return np.zeros_like(vec)
    return vec / nrm


def _rotation_to_local_z(direction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return O such that O @ n_hat = e_z.
    This matches the Mathematica use:
        omega . mat . Transpose[omega]
    """
    n = _normalize(direction, eps=eps)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    if np.linalg.norm(n) < eps:
        return np.eye(3, dtype=float)

    c = float(np.dot(n, ez))
    if c > 1.0 - eps:
        return np.eye(3, dtype=float)
    if c < -1.0 + eps:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

    v = np.cross(n, ez)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _direct_sum(mats: list[np.ndarray]) -> np.ndarray:
    rows = []
    for i, mi in enumerate(mats):
        row = []
        for j, mj in enumerate(mats):
            if i == j:
                row.append(mi)
            else:
                row.append(np.zeros((mi.shape[0], mj.shape[1]), dtype=complex))
        rows.append(row)
    return np.block(rows)


def _make_omega(dirs: np.ndarray) -> np.ndarray:
    return _direct_sum([_rotation_to_local_z(d) for d in dirs])


def _sigma2_metric(n_sublattices: int) -> np.ndarray:
    """
    Mathematica:
        metric = KroneckerProduct[IdentityMatrix[Length[R]/2], PauliMatrix[2]]
    where PauliMatrix[2] = [[0,-I],[I,0]]
    """
    sigma2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    return np.kron(np.eye(n_sublattices, dtype=complex), sigma2)


# ============================================================
# Symmetry data copied from the Mathematica notebook
# ============================================================

_ROT_AXES = np.array(
    [
        [np.sqrt(2.0 / 3.0) - 2.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0) - 1.0 / np.sqrt(3.0), np.sqrt(3.0)],
        [-(2.0 / np.sqrt(3.0)), -np.sqrt(2.0 / 3.0) - 1.0 / np.sqrt(3.0), np.sqrt(3.0)],
        [-np.sqrt(2.0 / 3.0) - 2.0 / np.sqrt(3.0), -(1.0 / np.sqrt(3.0)), np.sqrt(3.0)],
    ],
    dtype=float,
).T
_ROT_AXES_INV = np.linalg.inv(_ROT_AXES)

_RGS: Dict[str, np.ndarray] = {
    "g1": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "g2": np.array([[0, -1, -1], [1, -1, 0], [0, 0, 1]], dtype=float),
    "g3": np.array([[-1, 1, -1], [-1, 0, -1], [0, 0, 1]], dtype=float),
    "g4": np.array([[1, -1, 1], [1, 0, 1], [0, 0, -1]], dtype=float),
    "g5": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float),
    "g6": np.array([[0, 1, 1], [-1, 1, 0], [0, 0, -1]], dtype=float),
}


def _og_from_rg(rg: np.ndarray) -> np.ndarray:
    og_polar = _ROT_AXES_INV @ rg @ _ROT_AXES
    return np.linalg.det(rg) * og_polar


# ============================================================
# Exchange matrices from notebook-style parameters
# ============================================================

def _j_matrices_from_params(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    j1 = float(p.get("j1", 0.0))
    j2 = float(p.get("j2", 0.0))
    j3 = float(p.get("j3", 0.0))
    a = float(p.get("A", 0.0))

    kx = float(p.get("kx", 0.0))
    ky = float(p.get("ky", 0.0))
    kz = float(p.get("kz", 0.0))
    gxy = float(p.get("gxy", 0.0))
    gxz = float(p.get("gxz", 0.0))
    gyz = float(p.get("gyz", 0.0))
    d = float(p.get("d", 0.0))

    # J0
    j0m = np.array(
        [
            [0.0, a, 0.0],
            [a, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    # J1
    j1m = np.array(
        [
            [j1 + kx, gxy, gxz],
            [gxy, j1 + ky, gyz],
            [gxz, gyz, j1 + kz],
        ],
        dtype=float,
    )

    # J2
    j2m = np.array(
        [
            [j2, d, -d],
            [-d, j2, d],
            [d, -d, j2],
        ],
        dtype=float,
    )

    # J3
    j3m = np.eye(3, dtype=float) * j3

    def _mat_from_prefix(prefix: str) -> np.ndarray:
        return np.array(
            [
                [float(p.get(f"{prefix}xx", 0.0)), float(p.get(f"{prefix}xy", 0.0)), float(p.get(f"{prefix}xz", 0.0))],
                [float(p.get(f"{prefix}yx", p.get(f"{prefix}xy", 0.0))), float(p.get(f"{prefix}yy", 0.0)), float(p.get(f"{prefix}yz", 0.0))],
                [float(p.get(f"{prefix}zx", p.get(f"{prefix}xz", 0.0))), float(p.get(f"{prefix}zy", p.get(f"{prefix}yz", 0.0))), float(p.get(f"{prefix}zz", 0.0))],
            ],
            dtype=float,
        )

    return {
        "J0": j0m,
        "J1": j1m,
        "J2": j2m,
        "J3": j3m,
        "J1c": _mat_from_prefix("J1c"),
        "J2c": _mat_from_prefix("J2c"),
        "J3c": _mat_from_prefix("J3c"),
    }


# ============================================================
# totalJk construction
# ============================================================

def _phase_from_coords(k: np.ndarray, r1: Tuple[float, float, float], r2: Tuple[float, float, float]) -> complex:
    dr = np.asarray(r1, dtype=float) - np.asarray(r2, dtype=float)
    return np.exp(1j * np.dot(k, dr))


def _build_total_jk_spin(k: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """
    Build the 6x6 spin-component matrix totalJk in the notebook convention:
        (A_x, A_y, A_z, B_x, B_y, B_z)
    """
    mats = _j_matrices_from_params(p)
    total = np.zeros((6, 6), dtype=complex)

    families = {
        "J0": [
            (0, 0, "g1", (0, 0, 0), (0, 0, 0)),
            (1, 1, "g6", (1, 1, -2), (1, 1, -2)),
        ],
        "J1": [
            (0, 1, "g1", (0, 0, 0), (0, 0, 0)),
            (0, 1, "g3", (-2, 0, 0), (-2, -1, 0)),
            (0, 1, "g2", (-2, -2, 0), (-1, -2, 0)),
        ],
        "J2": [
            (0, 0, "g1", (0, 0, 0), (1, 0, 0)),
            (0, 0, "g3", (-2, 0, 0), (-2, 1, 0)),
            (0, 0, "g2", (-2, -2, 0), (-3, -3, 0)),
            (1, 1, "g6", (1, 1, -2), (2, 2, -2)),
            (1, 1, "g5", (-1, -1, -2), (-2, -1, -2)),
            (1, 1, "g4", (1, -1, -2), (1, -2, -2)),
        ],
        "J3": [
            (0, 1, "g1", (0, 0, 0), (1, 1, 0)),
            (0, 1, "g3", (-2, 0, 0), (-3, -1, 0)),
            (0, 1, "g2", (-2, -2, 0), (-1, -3, 0)),
        ],
        "J1c": [
            (0, 1, "g1", (0, 0, 0), (1, 0, -1)),
        ],
        "J2c": [
            (0, 1, "g1", (0, 0, 0), (0, 0, 1)),
            (0, 1, "g3", (-2, 0, 0), (-3, -1, 1)),
            (0, 1, "g2", (-2, -2, 0), (-2, -3, 1)),
        ],
        "J3c": [
            (0, 0, "g1", (0, 0, 0), (0, 0, 1)),
            (0, 0, "g3", (-2, 0, 0), (-3, 0, 1)),
            (0, 0, "g2", (-2, -2, 0), (-3, -3, 1)),
            (1, 1, "g6", (1, 1, -2), (2, 2, -3)),
            (1, 1, "g5", (-1, -1, -2), (-1, -1, -3)),
            (1, 1, "g4", (1, -1, -2), (2, -1, -3)),
        ],
    }

    for fam, entries in families.items():
        cur = [[np.zeros((3, 3), dtype=complex) for _ in range(2)] for _ in range(2)]
        for row, col, gname, r1, r2 in entries:
            rg = _RGS[gname]
            og = _og_from_rg(rg)
            curj = og.T @ mats[fam] @ og
            cur[row][col] += _phase_from_coords(k, r1, r2) * curj

        blocks = [[None, None], [None, None]]
        for row in range(2):
            for col in range(2):
                blocks[row][col] = 0.5 * (cur[row][col] + np.conjugate(cur[col][row].T))
        total += np.block(blocks)

    return 0.5 * (total + np.conjugate(total.T))


# ============================================================
# Notebook-literal reduced R(q) and metricR(q)
# ============================================================

def _build_reduced_R(model: SpinExchangeModel, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Literal translation of the notebook LSWT block:

      localtildeJ = omega . tildeJq . omega^T
      normalizedlocaltildeJq = normalizer . localtildeJ . normalizer
      chempot = DirectSum[(coeff_alpha * Diag[-1,-1,0])_alpha]
      R(q) = 2 (normalizedlocaltildeJq(q) + chempot)[Ridxs,Ridxs]

    Returns:
      R_reduced, metric, zeemanTerm
    """
    p = model.exchanges
    spin = float(p.get("S", 0.5))

    dirs_raw = p.get("dirs", ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    dirs = np.asarray(dirs_raw, dtype=float)
    if dirs.shape != (2, 3):
        raise ValueError("exchanges['dirs'] must have shape (2,3)")

    dirs_norm = np.vstack([_normalize(d) for d in dirs])

    total_jk = _build_total_jk_spin(q, p)
    total_j0 = _build_total_jk_spin(np.zeros(3, dtype=float), p)

    omega = _make_omega(dirs_norm)

    local_tilde_j = omega @ total_jk @ omega.T
    local_tilde_j0 = omega @ total_j0 @ omega.T

    normalizer = np.kron(np.diag(np.sqrt(np.full(2, spin))), np.eye(3, dtype=complex))

    normalized_local_tilde_j = normalizer @ local_tilde_j @ normalizer
    normalized_local_tilde_j0 = normalizer @ local_tilde_j0 @ normalizer

    # notebook:
    # szIdxs = Range[3, Length[normalizedlocaltildeJq[0,0,0]], 3]
    # in 0-based python: 2, 5
    sz_idxs = np.array([2, 5], dtype=int)

    # notebook:
    # chempot = DirectSum@((#*DiagonalMatrix[{-1,-1,0}]) & /@
    #      (Total /@ Outer[(normalizer[[#2,#2]]/normalizer[[#1,#1]])
    #       normalizedlocaltildeJq[0,0,0][[#1,#2]] &, szIdxs, szIdxs]));
    coeffs = []
    for i in sz_idxs:
        coeff = 0.0 + 0.0j
        for j in sz_idxs:
            coeff += (normalizer[j, j] / normalizer[i, i]) * normalized_local_tilde_j0[i, j]
        coeffs.append(coeff)

    chempot = _direct_sum([coeff * np.diag([-1.0, -1.0, 0.0]) for coeff in coeffs])

    # notebook:
    # Ridxs = Flatten[({-2,-1} + 3*#) & /@ Range[Length[szIdxs]]]
    # For two sublattices, 1-based gives {1,2,4,5}; 0-based -> {0,1,3,4}
    ridxs = np.array([0, 1, 3, 4], dtype=int)

    R_reduced = 2.0 * (normalized_local_tilde_j + chempot)[np.ix_(ridxs, ridxs)]

    # notebook metric:
    metric = _sigma2_metric(len(R_reduced) // 2)

    # reduced-space Zeeman term, matching the notebook's later use
    # zeemanTerm = field_scale * KroneckerProduct[DiagonalMatrix[(Normalize /@ dirs) . bfield], IdentityMatrix[2]]
    bfield = np.asarray(model.magnetic_field_xyz, dtype=float)
    field_scale = float(p.get("field_scale", 1.0))
    zeeman_term = field_scale * np.kron(np.diag(dirs_norm @ bfield), np.eye(2, dtype=complex))

    return R_reduced, metric, zeeman_term


def build_magnon_bilinear(
    model: SpinExchangeModel,
    kvec: Iterable[float],
) -> np.ndarray:
    """
    Return the notebook object

        metricR(q) = metric @ (R(q) + zeemanTerm)

    exactly as in Mathematica.

    IMPORTANT:
    metricR is generally non-Hermitian; do NOT symmetrize it.
    """
    q = np.asarray(tuple(kvec), dtype=float)
    if q.shape != (3,):
        raise ValueError("kvec must contain three components")

    R_reduced, metric, zeeman_term = _build_reduced_R(model, q)
    metricR = metric @ (R_reduced + zeeman_term)
    return np.asarray(metricR, dtype=complex)


def paraunitary_diagonalize(
    bilinear: np.ndarray,
    atol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Follow the Mathematica notebook literally:
    diagonalize metricR directly, sort by Re[eigenvalue],
    and keep the positive-frequency sector.

    Since metricR is non-Hermitian, use np.linalg.eig, not eigh.
    """
    bilinear = np.asarray(bilinear, dtype=complex)

    vals, vecs = np.linalg.eig(bilinear)
    vals = np.real_if_close(vals, tol=1000)

    order = np.argsort(np.real(vals))
    vals = np.real(vals[order])
    vecs = vecs[:, order]

    pos = np.where(vals > atol)[0]
    if len(pos) < 2:
        raise RuntimeError(
            "Could not extract two positive magnon modes; "
            "state may be unstable or parameters inconsistent."
        )

    vals_pos = vals[pos[:2]]
    vecs_pos = vecs[:, pos[:2]]
    return np.asarray(vals_pos), vecs_pos


def solve_band_structure(
    model: SpinExchangeModel,
    kpoints: np.ndarray,
) -> np.ndarray:
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("kpoints must have shape (N,3)")

    bands = np.empty((kpoints.shape[0], 2), dtype=float)
    for i, k in enumerate(kpoints):
        metricR = build_magnon_bilinear(model, k)
        vals, _ = paraunitary_diagonalize(metricR)
        bands[i, :] = np.sort(vals)
    return bands


# ============================================================
# Plotting helpers
# ============================================================

def _set_academic_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "figure.facecolor": "#fcfcfd",
            "axes.facecolor": "#fcfcfd",
            "savefig.facecolor": "#fcfcfd",
        }
    )


def _build_kpath(
    labels_to_coords: Dict[str, np.ndarray],
    path: list[str],
    points_per_segment: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pts = np.asarray([labels_to_coords[label] for label in path], dtype=float)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s_nodes = s_nodes / s_nodes[-1]

    s_vals: list[float] = []
    k_vals: list[np.ndarray] = []
    for i in range(len(path) - 1):
        start = pts[i]
        end = pts[i + 1]
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
        seg_k = start[None, :] + (end - start)[None, :] * t[:, None]
        seg_s = s_nodes[i] + (s_nodes[i + 1] - s_nodes[i]) * t
        s_vals.extend(seg_s.tolist())
        k_vals.extend(seg_k.tolist())

    s_vals.append(1.0)
    k_vals.append(pts[-1])
    return np.asarray(s_vals), np.asarray(k_vals), s_nodes, path


class MagnonLSWT:
    def __init__(self, model: SpinExchangeModel):
        self.model = model

    def build_magnon_bilinear(self, kvec: Iterable[float]) -> np.ndarray:
        return build_magnon_bilinear(self.model, kvec)

    def paraunitary_diagonalize(self, bilinear: np.ndarray, atol: float = 1e-9):
        return paraunitary_diagonalize(bilinear, atol=atol)

    def solve_band_structure(self, kpoints: np.ndarray) -> np.ndarray:
        return solve_band_structure(self.model, kpoints)


    def _band_on_plot_coords(self, kx: float, ky: float, bmat: np.ndarray, choose: str) -> float:
        q = bmat @ np.array([kx, ky], dtype=float)
        vals, _ = self.paraunitary_diagonalize(
            self.build_magnon_bilinear((q[0], q[1], 0.0))
        )
        vals = np.sort(vals)
        return float(vals[-1] if choose == "upper" else vals[0])


    def plot_band_cut_and_contour_pdf(
        self,
        out_pdf: Path,
        nk_contour: int = 121,
        choose: str = "lower",
        points_per_segment: int = 80,
    ) -> None:
        _set_academic_plot_style()

        labels_to_coords = {
            "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 0.0]),
            "G": np.array([0.0, 0.0, 0.0]),
            "M": np.array([np.pi, 0.0, 0.0]),
        }
        path = ["K", "G", "M", "K"]
        s_vals, k_vals, s_nodes, tick_labels = _build_kpath(
            labels_to_coords, path, points_per_segment=points_per_segment
        )
        bands_cut = self.solve_band_structure(k_vals)

        # same contour-style coordinates as before
        bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
        inv_bmat = np.linalg.inv(bmat)

        kx_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_contour)
        ky_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_contour)
        z = np.empty((nk_contour, nk_contour), dtype=float)

        for i, kx in enumerate(kx_grid):
            for j, ky in enumerate(ky_grid):
                z[j, i] = self._band_on_plot_coords(float(kx), float(ky), bmat, choose)
                q = bmat @ np.array([kx, ky], dtype=float)
                vals, _ = self.paraunitary_diagonalize(
                    self.build_magnon_bilinear((q[0], q[1], 0.0))
                )
                vals = np.sort(vals)
                z[j, i] = float(vals[-1] if choose == "upper" else vals[0])

        bz_q = (2.0 * np.pi / 3.0) * np.array(
            [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]],
            dtype=float,
        )
        bz_plot = bz_q @ inv_bmat.T
        gamma_pt = np.array([0.0, 0.0])
        k_pt = inv_bmat @ np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
        m_pt = inv_bmat @ np.array([np.pi, 0.0])

        fig, (ax_cut, ax_map) = plt.subplots(
            1,
            2,
            figsize=(11.2, 4.4),
            gridspec_kw={"width_ratios": [1.05, 1.25]},
            constrained_layout=True,
        )

        line_colors = ["#d2f413", "#dd0f4c"]
        for ib in range(bands_cut.shape[1]):
            ax_cut.plot(s_vals, bands_cut[:, ib], color=line_colors[ib % len(line_colors)], lw=2.1)
        for s in s_nodes:
            ax_cut.axvline(s, color="#94a3b8", ls="--", lw=0.8)

        y_min = float(np.min(bands_cut))
        y_max = float(np.max(bands_cut))
        y_pad = 0.08 * max(y_max - y_min, 1e-6)
        ax_cut.set_xlim(0.0, 1.0)
        ax_cut.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_cut.set_ylabel(r"$E(k)$")
        ax_cut.set_xticks(s_nodes)
        ax_cut.set_xticklabels(tick_labels)
        ax_cut.set_title("Band Cut")

        mesh = ax_map.imshow(
            z,
            extent=[kx_grid[0], kx_grid[-1], ky_grid[0], ky_grid[-1]],
            origin="lower",
            interpolation="bicubic",
            cmap="magma",
            aspect="auto",
        )
        ax_map.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.8)
        ax_map.scatter(
            [gamma_pt[0], k_pt[0], m_pt[0]],
            [gamma_pt[1], k_pt[1], m_pt[1]],
            c="#d1d5df",
            s=22,
            zorder=3,
        )
        ax_map.text(gamma_pt[0], gamma_pt[1], r"$\Gamma$", c="#d1d5df", fontsize=10, va="bottom")
        ax_map.text(k_pt[0], k_pt[1], r"$K$", c="#d1d5df", fontsize=10, va="bottom")
        ax_map.text(m_pt[0], m_pt[1], r"$M$", c="#d1d5df", fontsize=10, va="bottom")
        ax_map.set_xlabel(r"$k_x$")
        ax_map.set_ylabel(r"$k_y$")
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_title(f"2D Band Map ({choose})")
        cbar = fig.colorbar(mesh, ax=ax_map, pad=0.01)
        cbar.set_label(r"$E(k)$")

        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)


# ============================================================
# Main for the target parameter set
# ============================================================

def __main__() -> None:
    """
    Target parameter set:
      phi = 5 pi / 4
      J = cos(phi), K = sin(phi)
      Gamma = -0.5
      dirs = ([111],[111])

    For the field case, the notebook convention is subtle.
    Start with field_sign = +1.0.
    If you still do not see the expected structure, flip to -1.0.
    """
    n111 = np.array([1.0, 1.0, 1.0], dtype=float)
    n111 /= np.linalg.norm(n111)

    field_sign = +1.0

    model = SpinExchangeModel(
        exchanges={
            "S": 0.5,
            "A": 0.0,
            "j1": float(np.cos(5.0 * np.pi / 4.0)),
            "kx": 0.0,
            "ky": 0.0,
            "kz": float(np.sin(5.0 * np.pi / 4.0)),
            "gxy": -0.5,
            "gxz": 0.0,
            "gyz": 0.0,
            "j2": 0.0,
            "d": 0.00,
            "j3": 0.0,
            "dirs": (n111, n111),
            # In this script, field_scale=1 means the field enters as h directly.
            "field_scale": 1.0,
        },
        magnetic_field_xyz=field_sign * 0.0 * n111,
        symmetry="C3i",
    )

    solver = MagnonLSWT(model)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    band_panel_pdf = out_dir / "band_cut_contour_notebook_literal.pdf"

    solver.plot_band_cut_and_contour_pdf(
        band_panel_pdf,
        nk_contour=121,
        choose="lower",
    )

    labels_to_coords = {
        "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 0.0]),
        "G": np.array([0.0, 0.0, 0.0]),
        "M": np.array([np.pi, 0.0, 0.0]),
    }
    path = ["K", "G", "M", "K"]
    s_vals, k_vals, _, _ = _build_kpath(labels_to_coords, path, points_per_segment=80)

    bands_cut = solver.solve_band_structure(k_vals)
    sorted_bands_cut = np.sort(bands_cut, axis=1)
    cut_gap = sorted_bands_cut[:, 1] - sorted_bands_cut[:, 0]
    gap_idx = int(np.argmin(cut_gap))
    min_gap_cut = float(cut_gap[gap_idx])
    min_gap_s = float(s_vals[gap_idx])
    min_gap_k = k_vals[gap_idx]

    nk_gap = 81
    q1_grid = np.linspace(0.0, 2.0 * np.pi, nk_gap, endpoint=False)
    q2_grid = np.linspace(0.0, 2.0 * np.pi, nk_gap, endpoint=False)

    min_gap_bz = np.inf
    min_gap_point = None
    for q1 in q1_grid:
        for q2 in q2_grid:
            vals, _ = solver.paraunitary_diagonalize(
                solver.build_magnon_bilinear((float(q1), float(q2), 0.0))
            )
            vals = np.sort(vals)
            gap_here = float(vals[1] - vals[0])
            if gap_here < min_gap_bz:
                min_gap_bz = gap_here
                min_gap_point = (float(q1), float(q2), 0.0)

    print(f"Saved cut+contour PDF: {band_panel_pdf}")
    print(f"Minimum gap on K-G-M-K cut: {min_gap_cut:.6f} (at s={min_gap_s:.6f}, k={min_gap_k})")
    print(f"Minimum direct gap on full 2D BZ: {min_gap_bz:.6f} (at q={min_gap_point})")


if __name__ == "__main__":
    __main__()
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import ticker
from io import BytesIO, StringIO
from datetime import datetime

import sys
from pathlib import Path

# Support running from source layout without requiring editable install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from topocompass import SpinExchangeModel
from topocompass.core import MagnonLSWT


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    }
)


def _build_kpath(points_per_segment: int, path_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels_to_coords = {
        "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 0.0]),
        "G": np.array([0.0, 0.0, 0.0]),
        "M": np.array([np.pi, 0.0, 0.0]),
        "Mp": np.array([0.0, np.pi, 0.0]),
        "Mpp": np.array([-np.pi, np.pi, 0.0]),
        "Kp": np.array([-2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0, 0.0]),
    }
    if path_mode == "K-G-M-K":
        path = ["K", "G", "M", "K"]
    else:
        # Extended high-symmetry route for C3-broken cases.
        path = ["K", "G", "M", "G", "Mp", "G", "Mpp", "G", "Kp"]

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
        s_seg = s_nodes[i] + (s_nodes[i + 1] - s_nodes[i]) * t
        k_seg = start[None, :] + (end - start)[None, :] * t[:, None]
        s_vals.extend(s_seg.tolist())
        k_vals.extend(k_seg.tolist())

    s_vals.append(1.0)
    k_vals.append(pts[-1])
    display_labels = ["M'" if p == "Mp" else "M''" if p == "Mpp" else "K'" if p == "Kp" else p for p in path]
    return np.asarray(s_vals), np.asarray(k_vals), s_nodes, display_labels


def _matrix_to_csv_bytes(data: np.ndarray, header: str) -> bytes:
    buf = StringIO()
    np.savetxt(buf, data, delimiter=",", header=header, comments="")
    return buf.getvalue().encode("utf-8")


def _grid_to_csv_bytes(kx: np.ndarray, ky: np.ndarray, z: np.ndarray, z_name: str) -> bytes:
    kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing="xy")
    flat = np.column_stack([kx_mesh.ravel(), ky_mesh.ravel(), z.ravel()])
    return _matrix_to_csv_bytes(flat, f"kx,ky,{z_name}")


def _apply_pi_ticks(ax: plt.Axes) -> None:
    ticks = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float) * np.pi
    labels = [
        r"$-2\pi$",
        r"$-\pi$",
        r"$0$",
        r"$\pi$",
        r"$2\pi$",
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def _figure_to_pdf_bytes(fig: plt.Figure) -> bytes:
    fig_face = fig.get_facecolor()
    axes_state = []

    for ax in fig.axes:
        state = {
            "ax": ax,
            "facecolor": ax.get_facecolor(),
            "title_color": ax.title.get_color(),
            "xlabel_color": ax.xaxis.label.get_color(),
            "ylabel_color": ax.yaxis.label.get_color(),
            "x_tick_colors": [lbl.get_color() for lbl in ax.get_xticklabels()],
            "y_tick_colors": [lbl.get_color() for lbl in ax.get_yticklabels()],
            "spine_colors": {name: spine.get_edgecolor() for name, spine in ax.spines.items()},
            "gridlines": [(ln, ln.get_color(), ln.get_alpha()) for ln in (ax.get_xgridlines() + ax.get_ygridlines())],
            "texts": [],
        }
        for txt in ax.texts:
            patch = txt.get_bbox_patch()
            state["texts"].append(
                {
                    "text": txt,
                    "color": txt.get_color(),
                    "bbox_fc": patch.get_facecolor() if patch is not None else None,
                    "bbox_ec": patch.get_edgecolor() if patch is not None else None,
                }
            )
        axes_state.append(state)

    try:
        fig.patch.set_facecolor("white")
        for state in axes_state:
            ax = state["ax"]
            ax.set_facecolor("white")
            ax.title.set_color("black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.tick_params(colors="black")
            for spine in ax.spines.values():
                spine.set_color("black")
            for line, _color, _alpha in state["gridlines"]:
                line.set_color("#9ca3af")
                line.set_alpha(0.5)
            for tstate in state["texts"]:
                txt = tstate["text"]
                txt.set_color("black")
                patch = txt.get_bbox_patch()
                if patch is not None:
                    patch.set_facecolor("white")
                    patch.set_edgecolor("black")

        buf = BytesIO()
        # PDF is vector-based; dpi is set high for rasterized artists if present.
        fig.savefig(buf, format="pdf", bbox_inches="tight", dpi=600, facecolor="white")
        buf.seek(0)
        return buf.getvalue()
    finally:
        fig.patch.set_facecolor(fig_face)
        for state in axes_state:
            ax = state["ax"]
            ax.set_facecolor(state["facecolor"])
            ax.title.set_color(state["title_color"])
            ax.xaxis.label.set_color(state["xlabel_color"])
            ax.yaxis.label.set_color(state["ylabel_color"])
            for lbl, color in zip(ax.get_xticklabels(), state["x_tick_colors"]):
                lbl.set_color(color)
            for lbl, color in zip(ax.get_yticklabels(), state["y_tick_colors"]):
                lbl.set_color(color)
            for name, spine in ax.spines.items():
                spine.set_color(state["spine_colors"][name])
            for line, color, alpha in state["gridlines"]:
                line.set_color(color)
                line.set_alpha(alpha)
            for tstate in state["texts"]:
                txt = tstate["text"]
                txt.set_color(tstate["color"])
                patch = txt.get_bbox_patch()
                if patch is not None and tstate["bbox_fc"] is not None and tstate["bbox_ec"] is not None:
                    patch.set_facecolor(tstate["bbox_fc"])
                    patch.set_edgecolor(tstate["bbox_ec"])


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()


def _plot_band_cut(s_vals: np.ndarray, bands: np.ndarray, s_nodes: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    colors = ["#facc15", "#22d3ee", "#f472b6", "#86efac"]
    gap_text = ""
    if bands.shape[1] >= 2:
        bands_sorted = np.sort(bands, axis=1)
        gap_vals = bands_sorted[:, 1] - bands_sorted[:, 0]
        min_idx = int(np.argmin(gap_vals))
        min_gap = float(gap_vals[min_idx])
        gap_text = rf"$\Delta_{{\mathrm{{min}}}}={min_gap:.4f}$"

    for ib in range(bands.shape[1]):
        ax.plot(
            s_vals,
            bands[:, ib],
            color=colors[ib % len(colors)],
            lw=3.0,
            alpha=0.98,
            solid_capstyle="round",
        )
    for s in s_nodes:
        ax.axvline(s, color="#e2e8f0", ls="--", lw=1.0, alpha=0.9)

    y_min = float(np.min(bands))
    y_max = float(np.max(bands))
    pad = 0.08 * max(y_max - y_min, 1e-6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(s_nodes)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title("Band Cut")
    if gap_text:
        ax.text(
            0.03,
            0.97,
            gap_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="white"),
        )
    ax.grid(axis="y", color="#ffffff", alpha=0.22, lw=0.8)
    for spine in ax.spines.values():
        spine.set_color("#f8fafc")
        spine.set_linewidth(1.0)
    return fig


def _plot_band_cut_scaled(
    s_vals: np.ndarray,
    bands: np.ndarray,
    s_nodes: np.ndarray,
    labels: list[str],
    zoom: float,
):
    fig = _plot_band_cut(s_vals, bands, s_nodes, labels)
    ax = fig.axes[0]

    bands_sorted = np.sort(bands, axis=1)
    gap_vals = bands_sorted[:, 1] - bands_sorted[:, 0]
    min_idx = int(np.argmin(gap_vals))
    center = 0.5 * (float(bands_sorted[min_idx, 0]) + float(bands_sorted[min_idx, 1]))
    half_span = float(np.max(np.abs(bands - center)))
    y_half = max(half_span / max(zoom, 1e-6), 1e-6)
    y_half *= 1.04
    ax.set_ylim(center - y_half, center + y_half)
    return fig


def _plot_band_contour(solver, nk: int, band_index: int):
    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)

    kx = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    ky = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    z = np.empty((nk, nk), dtype=float)
    for i, k1 in enumerate(kx):
        for j, k2 in enumerate(ky):
            q = bmat @ np.array([k1, k2], dtype=float)
            evals, _ = solver.paraunitary_diagonalize(solver.build_magnon_bilinear((q[0], q[1], 0.0)))
            evals = np.sort(evals)
            if band_index < 0 or band_index >= len(evals):
                raise ValueError("band_index out of range for contour plot")
            z[j, i] = float(evals[band_index])

    bz_q = (2.0 * np.pi / 3.0) * np.array(
        [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]], dtype=float
    )
    bz_plot = bz_q @ inv_bmat.T

    hs_q = {
        "G": np.array([0.0, 0.0], dtype=float),
        "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0], dtype=float),
        "K'": np.array([-2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0], dtype=float),
        "M": np.array([np.pi, 0.0], dtype=float),
        "M'": np.array([0.0, np.pi], dtype=float),
        "M''": np.array([-np.pi, np.pi], dtype=float),
    }
    hs_plot = {label: inv_bmat @ q for label, q in hs_q.items()}

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    im = ax.imshow(
        z,
        extent=[kx[0], kx[-1], ky[0], ky[-1]],
        origin="lower",
        interpolation="bicubic",
        cmap="magma",
        aspect="auto",
    )
    ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.9)
    marker_color = "#67e8f9"
    for label, xy in hs_plot.items():
        ax.scatter(xy[0], xy[1], s=22, c=marker_color, edgecolors="#ecfeff", linewidths=0.4, zorder=3)
        ax.text(
            float(xy[0]),
            float(xy[1]),
            label,
            color=marker_color,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, edgecolor="none"),
        )
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    _apply_pi_ticks(ax)
    band_label = "Lower" if band_index == 0 else "Upper"
    ax.set_title(f"Band Contour ({band_label})")
    fig.colorbar(im, ax=ax, label=r"$E(k)$")
    return fig, {"kx": kx, "ky": ky, "energy": z}


def _band_eigenvector_from_core(solver, kx: float, ky: float, band_index: int, atol: float = 1e-9) -> np.ndarray:
    metric_r = solver.build_magnon_bilinear((float(kx), float(ky), 0.0))
    vals, vecs = np.linalg.eig(metric_r)
    order = np.argsort(np.real(vals))
    vals = np.real(vals[order])
    vecs = vecs[:, order]

    pos = np.where(vals > atol)[0]
    if len(pos) < 2:
        raise RuntimeError("Could not extract two positive magnon modes.")
    if band_index < 0 or band_index > 1:
        raise ValueError("band_index must be 0 or 1.")

    vec = vecs[:, pos[band_index]]
    nrm = float(np.linalg.norm(vec))
    if nrm < atol:
        raise RuntimeError("Encountered near-zero eigenvector norm.")
    return vec / nrm


def _derive_berry_curvature_from_core(solver, grid_n: int, band_index: int, eps: float = 1e-12) -> dict[str, np.ndarray]:
    kx = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False)
    ky = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False)

    # Store all eigenvectors in a dense complex tensor for fast vectorized overlaps.
    probe = _band_eigenvector_from_core(solver, float(kx[0]), float(ky[0]), band_index)
    vec_dim = probe.shape[0]
    frames = np.empty((grid_n, grid_n, vec_dim), dtype=complex)
    frames[0, 0, :] = probe
    for i, kxi in enumerate(kx):
        for j, kyj in enumerate(ky):
            if i == 0 and j == 0:
                continue
            frames[i, j, :] = _band_eigenvector_from_core(solver, float(kxi), float(kyj), band_index)

    frames_x = np.roll(frames, -1, axis=0)
    frames_y = np.roll(frames, -1, axis=1)
    frames_xy = np.roll(frames_x, -1, axis=1)

    o1 = np.sum(np.conj(frames) * frames_x, axis=2)
    o2 = np.sum(np.conj(frames_x) * frames_xy, axis=2)
    o3 = np.sum(np.conj(frames_xy) * frames_y, axis=2)
    o4 = np.sum(np.conj(frames_y) * frames, axis=2)

    n1 = np.abs(o1)
    n2 = np.abs(o2)
    n3 = np.abs(o3)
    n4 = np.abs(o4)
    valid = (n1 >= eps) & (n2 >= eps) & (n3 >= eps) & (n4 >= eps)

    flux = np.zeros((grid_n, grid_n), dtype=float)
    prod = np.ones((grid_n, grid_n), dtype=complex)
    prod[valid] = (o1[valid] / n1[valid]) * (o2[valid] / n2[valid]) * (o3[valid] / n3[valid]) * (o4[valid] / n4[valid])
    flux[valid] = np.angle(prod[valid])

    dk = 2.0 * np.pi / grid_n
    curvature = flux / (dk * dk)
    return {"curvature": curvature, "flux": flux, "kx": kx, "ky": ky}


def _compute_chern_number(payload: dict[str, np.ndarray]) -> float:
    return float(np.sum(payload["flux"]) / (2.0 * np.pi))


def _compute_chern_number_fhs_honeycomb(
    solver,
    num: int,
    band_index: int,
    eps: float = 1e-12,
) -> float:
    """FHS Chern number on native periodic BZ discretization.

    Uses independent discretization ``num`` and gauge-invariant plaquette phases
    from Fukui-Hatsugai-Suzuki (cond-mat/0503172).
    """
    kx = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    ky = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)

    probe = _band_eigenvector_from_core(solver, float(kx[0]), float(ky[0]), band_index)
    vec_dim = probe.shape[0]
    frames = np.empty((num, num, vec_dim), dtype=complex)
    frames[0, 0, :] = probe
    for i, kxi in enumerate(kx):
        for j, kyj in enumerate(ky):
            if i == 0 and j == 0:
                continue
            frames[i, j, :] = _band_eigenvector_from_core(solver, float(kxi), float(kyj), band_index)

    frames_x = np.roll(frames, -1, axis=0)
    frames_y = np.roll(frames, -1, axis=1)
    frames_xy = np.roll(frames_x, -1, axis=1)

    o1 = np.sum(np.conj(frames) * frames_x, axis=2)
    o2 = np.sum(np.conj(frames_x) * frames_xy, axis=2)
    o3 = np.sum(np.conj(frames_xy) * frames_y, axis=2)
    o4 = np.sum(np.conj(frames_y) * frames, axis=2)

    n1 = np.abs(o1)
    n2 = np.abs(o2)
    n3 = np.abs(o3)
    n4 = np.abs(o4)
    valid = (n1 >= eps) & (n2 >= eps) & (n3 >= eps) & (n4 >= eps)

    prod = np.ones((num, num), dtype=complex)
    prod[valid] = (o1[valid] / n1[valid]) * (o2[valid] / n2[valid]) * (o3[valid] / n3[valid]) * (o4[valid] / n4[valid])

    flux = np.zeros((num, num), dtype=float)
    flux[valid] = np.angle(prod[valid])
    return float(np.sum(flux) / (2.0 * np.pi))


def _plot_berry_curvature(
    payload: dict[str, np.ndarray],
    nk: int,
    chern: float,
    band_index: int,
):
    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)

    curv_q = payload["curvature"]

    kx = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    ky = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    curv_plot = np.empty((nk, nk), dtype=float)
    for i, k1 in enumerate(kx):
        for j, k2 in enumerate(ky):
            q = bmat @ np.array([k1, k2], dtype=float)
            qx = (q[0] % (2.0 * np.pi)) / (2.0 * np.pi) * curv_q.shape[0]
            qy = (q[1] % (2.0 * np.pi)) / (2.0 * np.pi) * curv_q.shape[1]

            i0 = int(np.floor(qx)) % curv_q.shape[0]
            j0 = int(np.floor(qy)) % curv_q.shape[1]
            i1 = (i0 + 1) % curv_q.shape[0]
            j1 = (j0 + 1) % curv_q.shape[1]
            tx = qx - np.floor(qx)
            ty = qy - np.floor(qy)

            v00 = curv_q[i0, j0]
            v10 = curv_q[i1, j0]
            v01 = curv_q[i0, j1]
            v11 = curv_q[i1, j1]
            curv_plot[j, i] = (
                (1.0 - tx) * (1.0 - ty) * v00
                + tx * (1.0 - ty) * v10
                + (1.0 - tx) * ty * v01
                + tx * ty * v11
            )

    bz_q = (2.0 * np.pi / 3.0) * np.array(
        [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]], dtype=float
    )
    bz_plot = bz_q @ inv_bmat.T

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    im = ax.imshow(
        curv_plot,
        extent=[kx[0], kx[-1], ky[0], ky[-1]],
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-5.0,
        vmax=5.0,
        aspect="auto",
    )
    ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.9)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    _apply_pi_ticks(ax)
    band_label = "Lower" if band_index == 0 else "Upper"
    ax.set_title(f"Berry Curvature ({band_label})")
    ax.text(
        0.03,
        0.97,
        rf"$C_{{\mathrm{{band}}}}={chern:.3f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="white"),
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$\Omega(k)$")
    ticks, exponent, scale = _berry_integer_ticks_and_exponent(5.0)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_int_scaled(scale)))
    cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$", pad=6)
    return fig, {"kx": kx, "ky": ky, "curvature": curv_plot}


def _plot_band_contour_from_data(contour_data: dict[str, np.ndarray], band_index: int, zoom: float):
    kx = contour_data["kx"]
    ky = contour_data["ky"]
    z = contour_data["energy"]

    zmin = float(np.min(z))
    zmax = float(np.max(z))
    zmid = 0.5 * (zmin + zmax)
    zhalf = max(0.5 * (zmax - zmin) / max(zoom, 1e-6), 1e-9)

    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)
    bz_q = (2.0 * np.pi / 3.0) * np.array(
        [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]], dtype=float
    )
    bz_plot = bz_q @ inv_bmat.T

    hs_q = {
        "G": np.array([0.0, 0.0], dtype=float),
        "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0], dtype=float),
        "K'": np.array([-2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0], dtype=float),
        "M": np.array([np.pi, 0.0], dtype=float),
        "M'": np.array([0.0, np.pi], dtype=float),
        "M''": np.array([-np.pi, np.pi], dtype=float),
    }
    hs_plot = {label: inv_bmat @ q for label, q in hs_q.items()}

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    im = ax.imshow(
        z,
        extent=[float(kx[0]), float(kx[-1]), float(ky[0]), float(ky[-1])],
        origin="lower",
        interpolation="bicubic",
        cmap="magma",
        vmin=zmid - zhalf,
        vmax=zmid + zhalf,
        aspect="auto",
    )
    ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.9)
    marker_color = "#67e8f9"
    for label, xy in hs_plot.items():
        ax.scatter(xy[0], xy[1], s=22, c=marker_color, edgecolors="#ecfeff", linewidths=0.4, zorder=3)
        ax.text(
            float(xy[0]),
            float(xy[1]),
            label,
            color=marker_color,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, edgecolor="none"),
        )
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    _apply_pi_ticks(ax)
    band_label = "Lower" if band_index == 0 else "Upper"
    ax.set_title(f"Band Contour ({band_label})")
    fig.colorbar(im, ax=ax, label=r"$E(k)$")
    return fig


def _plot_berry_curvature_from_data(
    berry_data: dict[str, np.ndarray],
    chern: float,
    band_index: int,
    zoom: float,
):
    kx = berry_data["kx"]
    ky = berry_data["ky"]
    curv_plot = berry_data["curvature"]

    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)
    bz_q = (2.0 * np.pi / 3.0) * np.array(
        [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]], dtype=float
    )
    bz_plot = bz_q @ inv_bmat.T

    half = max(5.0 / max(zoom, 1e-6), 1e-6)

    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    im = ax.imshow(
        curv_plot,
        extent=[float(kx[0]), float(kx[-1]), float(ky[0]), float(ky[-1])],
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-half,
        vmax=half,
        aspect="auto",
    )
    ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.9)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    _apply_pi_ticks(ax)
    band_label = "Lower" if band_index == 0 else "Upper"
    ax.set_title(f"Berry Curvature ({band_label})")
    ax.text(
        0.03,
        0.97,
        rf"$C_{{\mathrm{{band}}}}={chern:.3f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="white"),
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$\Omega(k)$")
    ticks, exponent, scale = _berry_integer_ticks_and_exponent(half)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_int_scaled(scale)))
    cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$", pad=6)
    return fig


def _scale_from_center_slider(ctrl: float) -> float:
    """Map centered slider control in [-1, 1] to zoom scale, with 0 -> 1."""
    return float(10.0 ** (ctrl))


def _format_sci_1sig(x: float, _pos: int) -> str:
    """Format ticks as scientific notation with 1 significant figure."""
    x = float(x)
    if np.isclose(x, 0.0):
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = float(np.round(x / (10.0 ** exponent), 0))
    if abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1

    if np.isclose(mantissa, 0.0):
        return r"$0$"
    return rf"${int(mantissa)}\times10^{{{exponent}}}$"


def _berry_integer_ticks_and_exponent(vmax: float) -> tuple[np.ndarray, int, float]:
    vmax = float(abs(vmax))
    if vmax < 1e-12:
        return np.array([-1.0, 0.0, 1.0]), 0, 1.0

    exponent = int(np.floor(np.log10(vmax)))
    scale = float(10.0 ** exponent)
    max_int = int(np.floor(vmax / scale))
    max_int = max(1, min(max_int, 9))
    ticks = np.arange(-max_int, max_int + 1, dtype=float) * scale
    return ticks, exponent, scale


def _format_int_scaled(scale: float):
    def _fmt(x: float, _pos: int) -> str:
        return f"{int(np.round(float(x) / scale))}"

    return _fmt


def _sync_slider_from_number(num_key: str, sld_key: str, smin: float, smax: float) -> None:
    val = float(st.session_state[num_key])
    st.session_state[sld_key] = float(np.clip(val, smin, smax))


def _sync_number_from_slider(num_key: str, sld_key: str) -> None:
    st.session_state[num_key] = float(st.session_state[sld_key])


def _number_with_slider(
    label: str,
    base_key: str,
    default: float,
    slider_min: float,
    slider_max: float,
    number_step: float = 0.01,
    number_format: str = "%.6f",
    slider_step: float = 0.01,
    number_min: float | None = None,
    number_max: float | None = None,
) -> float:
    num_key = f"{base_key}_num"
    sld_key = f"{base_key}_sld"

    if num_key not in st.session_state:
        st.session_state[num_key] = float(default)
    if sld_key not in st.session_state:
        st.session_state[sld_key] = float(np.clip(default, slider_min, slider_max))

    st.number_input(
        label,
        min_value=number_min,
        max_value=number_max,
        step=float(number_step),
        format=number_format,
        key=num_key,
        on_change=_sync_slider_from_number,
        args=(num_key, sld_key, float(slider_min), float(slider_max)),
    )
    st.slider(
        f"{label} slider",
        min_value=float(slider_min),
        max_value=float(slider_max),
        step=float(slider_step),
        key=sld_key,
        on_change=_sync_number_from_slider,
        args=(num_key, sld_key),
        label_visibility="collapsed",
    )
    return float(st.session_state[num_key])


def _angles_from_xyz_deg(x: float, y: float, z: float) -> tuple[float, float]:
    vec = np.array([x, y, z], dtype=float)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-12:
        return 54.7356103172, 45.0

    u = vec / nrm
    theta = float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))
    phi = float(np.degrees(np.arctan2(u[1], u[0])))
    return theta, phi


def _xyz_from_angles_deg(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    th = np.deg2rad(float(theta_deg))
    ph = np.deg2rad(float(phi_deg))
    x = float(np.sin(th) * np.cos(ph))
    y = float(np.sin(th) * np.sin(ph))
    z = float(np.cos(th))
    return x, y, z


def _set_dir_angles_from_xyz_state() -> None:
    x = float(st.session_state["dir_x_num"])
    y = float(st.session_state["dir_y_num"])
    z = float(st.session_state["dir_z_num"])
    theta, phi = _angles_from_xyz_deg(x, y, z)
    st.session_state["theta_num"] = float(np.clip(theta, 0.0, 180.0))
    st.session_state["theta_sld"] = float(np.clip(theta, 0.0, 180.0))
    st.session_state["phi_num"] = float(np.clip(phi, -180.0, 180.0))
    st.session_state["phi_sld"] = float(np.clip(phi, -180.0, 180.0))


def _set_xyz_from_dir_angles_state() -> None:
    theta = float(st.session_state["theta_num"])
    phi = float(st.session_state["phi_num"])
    x, y, z = _xyz_from_angles_deg(theta, phi)
    st.session_state["dir_x_num"] = x
    st.session_state["dir_y_num"] = y
    st.session_state["dir_z_num"] = z
    st.session_state["dir_x_sld"] = float(np.clip(x, -3.0, 3.0))
    st.session_state["dir_y_sld"] = float(np.clip(y, -3.0, 3.0))
    st.session_state["dir_z_sld"] = float(np.clip(z, -3.0, 3.0))


def _on_dir_number_change(axis: str) -> None:
    num_key = f"dir_{axis}_num"
    sld_key = f"dir_{axis}_sld"
    st.session_state[sld_key] = float(np.clip(st.session_state[num_key], -3.0, 3.0))
    _set_dir_angles_from_xyz_state()


def _on_dir_slider_change(axis: str) -> None:
    num_key = f"dir_{axis}_num"
    sld_key = f"dir_{axis}_sld"
    st.session_state[num_key] = float(st.session_state[sld_key])
    _set_dir_angles_from_xyz_state()


def _on_theta_number_change() -> None:
    st.session_state["theta_sld"] = float(np.clip(st.session_state["theta_num"], 0.0, 180.0))
    _set_xyz_from_dir_angles_state()


def _on_theta_slider_change() -> None:
    st.session_state["theta_num"] = float(st.session_state["theta_sld"])
    _set_xyz_from_dir_angles_state()


def _on_phi_number_change() -> None:
    st.session_state["phi_sld"] = float(np.clip(st.session_state["phi_num"], -180.0, 180.0))
    _set_xyz_from_dir_angles_state()


def _on_phi_slider_change() -> None:
    st.session_state["phi_num"] = float(st.session_state["phi_sld"])
    _set_xyz_from_dir_angles_state()


def _init_direction_state() -> None:
    if "dir_x_num" not in st.session_state:
        st.session_state["dir_x_num"] = 1.0
    if "dir_y_num" not in st.session_state:
        st.session_state["dir_y_num"] = 1.0
    if "dir_z_num" not in st.session_state:
        st.session_state["dir_z_num"] = 1.0
    if "dir_x_sld" not in st.session_state:
        st.session_state["dir_x_sld"] = float(np.clip(st.session_state["dir_x_num"], -3.0, 3.0))
    if "dir_y_sld" not in st.session_state:
        st.session_state["dir_y_sld"] = float(np.clip(st.session_state["dir_y_num"], -3.0, 3.0))
    if "dir_z_sld" not in st.session_state:
        st.session_state["dir_z_sld"] = float(np.clip(st.session_state["dir_z_num"], -3.0, 3.0))

    if "theta_num" not in st.session_state or "phi_num" not in st.session_state:
        theta, phi = _angles_from_xyz_deg(
            float(st.session_state["dir_x_num"]),
            float(st.session_state["dir_y_num"]),
            float(st.session_state["dir_z_num"]),
        )
        st.session_state["theta_num"] = float(np.clip(theta, 0.0, 180.0))
        st.session_state["phi_num"] = float(np.clip(phi, -180.0, 180.0))

    if "theta_sld" not in st.session_state:
        st.session_state["theta_sld"] = float(np.clip(st.session_state["theta_num"], 0.0, 180.0))
    if "phi_sld" not in st.session_state:
        st.session_state["phi_sld"] = float(np.clip(st.session_state["phi_num"], -180.0, 180.0))


def _reset_controls_to_defaults() -> None:
    defaults: dict[str, float] = {
        "spin_s_num": 0.5,
        "spin_s_sld": 0.5,
        "anis_a_num": 0.0,
        "anis_a_sld": 0.0,
        "j1_num": float(np.cos(5.0 * np.pi / 4.0)),
        "j1_sld": float(np.cos(5.0 * np.pi / 4.0)),
        "k_term_num": float(np.sin(5.0 * np.pi / 4.0)),
        "k_term_sld": float(np.sin(5.0 * np.pi / 4.0)),
        "gamma_num": -0.50,
        "gamma_sld": -0.50,
        "gamma_p_num": -0.0,
        "gamma_p_sld": -0.0,
        "d_term_num": 0.0,
        "d_term_sld": 0.0,
        "j2_num": 0.0,
        "j2_sld": 0.0,
        "j3_num": 0.0,
        "j3_sld": 0.0,
        "dir_x_num": 1.0,
        "dir_x_sld": 1.0,
        "dir_y_num": 1.0,
        "dir_y_sld": 1.0,
        "dir_z_num": 1.0,
        "dir_z_sld": 1.0,
        "b_strength_num": 4.0,
        "b_strength_sld": 4.0,
    }
    for k, v in defaults.items():
        st.session_state[k] = v

    theta, phi = _angles_from_xyz_deg(1.0, 1.0, 1.0)
    st.session_state["theta_num"] = theta
    st.session_state["theta_sld"] = theta
    st.session_state["phi_num"] = phi
    st.session_state["phi_sld"] = phi


st.set_page_config(page_title="Spin Wave Explorer", layout="wide")
st.title("Spin Wave Explorer")
st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: "Times New Roman", Times, serif !important;
    }
    .stApp { background-color: #000000; }
    [data-testid="stSidebar"] { background-color: #000000; }
    [data-testid="stAppViewContainer"] { color: #ffffff; }
    [data-testid="stSidebar"] * { color: #ffffff; }
    .app-lead {
        font-size: clamp(1.9rem, 2.7vw, 2.9rem);
        line-height: 1.6;
        font-weight: 300;
        margin: 0.2rem 0 0.9rem 0;
        color: #e5e7eb;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] label p,
    [data-testid="stNumberInput"] .stMarkdown {
        font-size: 0.98rem !important;
        line-height: 1.2 !important;
    }
    .stCaption {
        font-size: 1.2rem !important;
        line-height: 1.55 !important;
    }

    .fd-separator {
        border-left: 1px solid #6b7280;
        min-height: 6.2rem;
        margin: 0 auto;
    }
    @media (max-width: 768px) {
        .fd-separator {
            border-left: none;
            border-top: 1px solid #6b7280;
            min-height: 0;
            width: 100%;
            margin: 0.35rem 0;
        }
    }

    </style>
    <p class="app-lead">
    Automated generator and solver of magnon bilinears for generic spin-orbit coupled honeycomb Mott insulators
    with C3i-symmetry-allowed spin exchanges and arbitrary 3D magnetic field polarizations. Parameters below map
    directly to the Hamiltonian terms shown here.
    </p>
    """,
    unsafe_allow_html=True,
)
st.latex(
    r"""
    \begin{aligned}
    H =\;& \sum_{\langle ij \rangle_{\gamma}} \Big[
    J\,\mathbf{S}_i \cdot \mathbf{S}_j
    + K\,S_i^{\gamma} S_j^{\gamma}
    + \Gamma\,(S_i^{\alpha}S_j^{\beta}+S_i^{\beta}S_j^{\alpha})
    + \Gamma'\,(S_i^{\gamma}S_j^{\alpha}+S_i^{\alpha}S_j^{\gamma}
    +S_i^{\gamma}S_j^{\beta}+S_i^{\beta}S_j^{\gamma})
    \Big] \\
    &+ J_2 \sum_{\langle\!\langle ij \rangle\!\rangle} \mathbf{S}_i \cdot \mathbf{S}_j
    + D \sum_{\langle\!\langle ij \rangle\!\rangle} \hat{\mathbf D}\cdot(\mathbf{S}_i \times \mathbf{S}_j)
    + J_3 \sum_{\langle\!\langle\!\langle ij \rangle\!\rangle\!\rangle} \mathbf{S}_i \cdot \mathbf{S}_j 
    + A \sum_i \left(S_i^x S_i^y + S_i^y S_i^x\right)
    - h \sum_i \hat{\mathbf{n}} \cdot \mathbf{S}_i .
    \end{aligned}
    """
)
_last_updated = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {_last_updated}")

st.subheader("Model Parameters")

if st.button("Reset to reference defaults"):
    _reset_controls_to_defaults()
    st.rerun()

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    spin_s = _number_with_slider(
        r"$S$ (Spin)",
        "spin_s",
        default=0.5,
        slider_min=0.5,
        slider_max=6.0,
        number_step=0.5,
        number_format="%.1f",
        slider_step=0.5,
        number_min=0.5,
        number_max=6.0,
    )
with pc2:
    anis_a = _number_with_slider(r"$A$", "anis_a", default=0.0, slider_min=-3.0, slider_max=3.0)
with pc3:
    j1 = _number_with_slider(
        r"$J$",
        "j1",
        default=float(np.cos(5.0 * np.pi / 4.0)),
        slider_min=-3.0,
        slider_max=3.0,
    )
with pc4:
    k_term = _number_with_slider(
        r"$K$",
        "k_term",
        default=float(np.sin(5.0 * np.pi / 4.0)),
        slider_min=-3.0,
        slider_max=3.0,
    )

pc5, pc6, pc7, pc8 = st.columns(4)
with pc5:
    gamma = _number_with_slider(r"$\Gamma$", "gamma", default=-0.50, slider_min=-3.0, slider_max=3.0)
with pc6:
    gamma_p = _number_with_slider(r"$\Gamma'$", "gamma_p", default=-0.0, slider_min=-3.0, slider_max=3.0)
with pc7:
    d_term = _number_with_slider(r"$D$", "d_term", default=0.0, slider_min=-3.0, slider_max=3.0)
with pc8:
    j2 = _number_with_slider(r"$J_2$", "j2", default=0.0, slider_min=-3.0, slider_max=3.0)

pc9, _, _, _ = st.columns(4)
with pc9:
    j3 = _number_with_slider(r"$J_3$", "j3", default=0.0, slider_min=-3.0, slider_max=3.0)

st.subheader("Field and Direction")
st.caption(
    r"Direction can be set in two equivalent ways: Cartesian $[d_x,d_y,d_z]$ or angles $(\theta,\phi)$ with "
    r"$\hat{\mathbf n}=(\sin\theta\cos\phi,\,\sin\theta\sin\phi,\,\cos\theta)$. "
    r"The app keeps both parameterizations synchronized automatically, so you only need to edit one row. "
    r"The field magnitude is controlled by `h = field_strength`, i.e. $\mathbf h = h\,\hat{\mathbf n}$"
)
_init_direction_state()
fd_cols = st.columns([1, 1, 1, 0.06, 1, 1, 0.06, 1])
with fd_cols[0]:
    st.number_input(
        r"$d_x$",
        key="dir_x_num",
        format="%.6f",
        on_change=_on_dir_number_change,
        args=("x",),
    )
    st.slider(
        "dir_x slider",
        min_value=-3.0,
        max_value=3.0,
        step=0.01,
        key="dir_x_sld",
        on_change=_on_dir_slider_change,
        args=("x",),
        label_visibility="collapsed",
    )
with fd_cols[1]:
    st.number_input(
        r"$d_y$",
        key="dir_y_num",
        format="%.6f",
        on_change=_on_dir_number_change,
        args=("y",),
    )
    st.slider(
        "dir_y slider",
        min_value=-3.0,
        max_value=3.0,
        step=0.01,
        key="dir_y_sld",
        on_change=_on_dir_slider_change,
        args=("y",),
        label_visibility="collapsed",
    )
with fd_cols[2]:
    st.number_input(
        r"$d_z$",
        key="dir_z_num",
        format="%.6f",
        on_change=_on_dir_number_change,
        args=("z",),
    )
    st.slider(
        "dir_z slider",
        min_value=-3.0,
        max_value=3.0,
        step=0.01,
        key="dir_z_sld",
        on_change=_on_dir_slider_change,
        args=("z",),
        label_visibility="collapsed",
    )
with fd_cols[3]:
    st.markdown("<div class='fd-separator'></div>", unsafe_allow_html=True)
with fd_cols[4]:
    st.number_input(
        r"$\theta$ (deg)",
        min_value=0.0,
        max_value=180.0,
        key="theta_num",
        format="%.6f",
        on_change=_on_theta_number_change,
    )
    st.slider(
        "theta slider",
        min_value=0.0,
        max_value=180.0,
        step=0.1,
        key="theta_sld",
        on_change=_on_theta_slider_change,
        label_visibility="collapsed",
    )
with fd_cols[5]:
    st.number_input(
        r"$\phi$ (deg)",
        min_value=-180.0,
        max_value=180.0,
        key="phi_num",
        format="%.6f",
        on_change=_on_phi_number_change,
    )
    st.slider(
        "phi slider",
        min_value=-180.0,
        max_value=180.0,
        step=0.1,
        key="phi_sld",
        on_change=_on_phi_slider_change,
        label_visibility="collapsed",
    )
with fd_cols[6]:
    st.markdown("<div class='fd-separator'></div>", unsafe_allow_html=True)
with fd_cols[7]:
    b_strength = _number_with_slider(
        "field_strength",
        "b_strength",
        default=4.0,
        slider_min=0.0,
        slider_max=50.0,
        number_step=0.1,
        number_format="%.6f",
        slider_step=0.1,
        number_min=0.0,
        number_max=50.0,
    )

dir_x = float(st.session_state["dir_x_num"])
dir_y = float(st.session_state["dir_y_num"])
dir_z = float(st.session_state["dir_z_num"])

with st.expander("Effective values for next Run", expanded=False):
    st.caption("These are the exact UI values that will be sent to the solver when Run is clicked.")
    st.write(
        {
            "S": float(spin_s),
            "A": float(anis_a),
            "J": float(j1),
            "K": float(k_term),
            "Gamma": float(gamma),
            "Gamma_prime": float(gamma_p),
            "D": float(d_term),
            "J2": float(j2),
            "J3": float(j3),
            "dir_x": float(dir_x),
            "dir_y": float(dir_y),
            "dir_z": float(dir_z),
            "theta_deg": float(st.session_state["theta_num"]),
            "phi_deg": float(st.session_state["phi_num"]),
            "field_strength": float(b_strength),
        }
    )

st.subheader("Resolution")
with st.form("run_controls_form"):
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        path_pts = st.number_input("k-path points/segment", min_value=20, max_value=400, value=30, step=10)
    with r2:
        contour_nk = st.number_input("Band contour grid", min_value=20, max_value=401, value=30, step=10)
    with r3:
        chern_grid = st.number_input("Berry curvature grid", min_value=30, max_value=2000, value=30, step=10)
    with r4:
        chern_num = st.number_input("BZ discretization for Chern number (FHS)", min_value=4, max_value=201, value=8, step=1)

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        band_choice = st.selectbox(
            "Band Selection for Contours",
            options=["Lower positive band (index 0)", "Upper positive band (index 1)"],
            index=1,
        )
    with sel_col2:
        cut_path_mode = st.selectbox(
            "Band-cut path",
            options=[
                "K-G-M-K",
                "K-G-M-G-M'-G-M''-G-K'",
            ],
            index=0,
        )

    topo_band_index = 0 if band_choice.startswith("Lower") else 1
    run_clicked = st.form_submit_button("Run", type="primary")

if run_clicked:
    run_id = int(st.session_state.get("_run_id_counter", 0)) + 1
    st.session_state["_run_id_counter"] = run_id

    direction = np.array([dir_x, dir_y, dir_z], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-12:
        st.error("Field direction cannot be zero vector.")
        st.stop()
    direction_unit = direction / direction_norm
    field_vec = float(b_strength) * direction_unit

    pvals: dict[str, float | tuple[tuple[float, float, float], tuple[float, float, float]]] = {
        "S": float(spin_s),
        "A": float(anis_a),
        "j1": float(j1),
        "kx": 0.0,
        "ky": 0.0,
        "kz": float(k_term),
        "gxy": float(gamma),
        "gxz": float(gamma_p),
        "gyz": float(gamma_p),
        "j2": float(j2),
        "d": float(d_term),
        "j3": float(j3),
        "bfield_strength": float(b_strength),
    }
    # Keep field direction and local axes consistent as requested.
    pvals["dirs"] = (
        (float(direction_unit[0]), float(direction_unit[1]), float(direction_unit[2])),
        (float(direction_unit[0]), float(direction_unit[1]), float(direction_unit[2])),
    )

    model = SpinExchangeModel(
        exchanges=pvals,  # type: ignore[arg-type]
        magnetic_field_xyz=(float(field_vec[0]), float(field_vec[1]), float(field_vec[2])),
        symmetry="C3i",
    )
    solver = MagnonLSWT(model)

    try:
        with st.spinner("Computing band cut, contour, Berry curvature, and Chern number... this may take ~10 seconds or more for high resolutions."):
            s_vals, k_vals, s_nodes, labels = _build_kpath(int(path_pts), cut_path_mode)
            bands = solver.solve_band_structure(k_vals)
            sorted_bands = np.sort(bands, axis=1)
            min_gap = float(np.min(sorted_bands[:, 1] - sorted_bands[:, 0]))
            gapless_warning = min_gap < 1e-2
            chern_warning = min_gap < 1e-2
            fig_cut = _plot_band_cut(s_vals, bands, s_nodes, labels)
            fig_contour, contour_data = _plot_band_contour(solver, int(contour_nk), topo_band_index)
            payload = _derive_berry_curvature_from_core(
                solver,
                grid_n=int(chern_grid),
                band_index=topo_band_index,
            )
            chern = _compute_chern_number_fhs_honeycomb(
                solver,
                num=int(chern_num),
                band_index=topo_band_index,
            )
            fig_berry, berry_data = _plot_berry_curvature(
                payload,
                int(chern_grid),
                chern,
                topo_band_index,
            )

            cut_header = "s,kx,ky,band0,band1"
            cut_data = np.column_stack([s_vals, k_vals[:, 0], k_vals[:, 1], bands[:, 0], bands[:, 1]])
            csv_cut = _matrix_to_csv_bytes(cut_data, cut_header)
            csv_contour = _grid_to_csv_bytes(contour_data["kx"], contour_data["ky"], contour_data["energy"], "energy")
            csv_berry = _grid_to_csv_bytes(berry_data["kx"], berry_data["ky"], berry_data["curvature"], "curvature")

            st.session_state["last_results"] = {
                "run_id": run_id,
                "s_vals": s_vals,
                "bands": bands,
                "s_nodes": s_nodes,
                "labels": labels,
                "contour_data": contour_data,
                "berry_data": berry_data,
                "topo_band_index": topo_band_index,
                "gapless_warning": gapless_warning,
                "chern_warning": chern_warning,
                "chern": chern,
                "csv_cut": csv_cut,
                "csv_contour": csv_contour,
                "csv_berry": csv_berry,
            }
    except RuntimeError as exc:
        st.error("Solver failed for this parameter set.")
        st.caption(str(exc))
        st.info(
            "Try increasing field_strength, changing exchange parameters, or reducing contour/chern grid to scan a nearby stable region."
        )
        st.stop()

if "last_results" in st.session_state:
    res = st.session_state["last_results"]

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2], vertical_alignment="top")
    with c1:
        cut_zoom_ctrl = float(st.session_state.get("cut_zoom_ctrl", 0.0))
        cut_zoom = _scale_from_center_slider(cut_zoom_ctrl)
        fig_cut = _plot_band_cut_scaled(
            res["s_vals"],
            res["bands"],
            res["s_nodes"],
            res["labels"],
            float(cut_zoom),
        )
        pdf_cut = _figure_to_pdf_bytes(fig_cut)
        st.pyplot(fig_cut, clear_figure=True)
        st.slider(
            "Band cut y-scale",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="cut_zoom_ctrl",
        )
        st.caption(f"Band cut scale = {cut_zoom:.3g}")
        if res["gapless_warning"]:
            st.warning("Potentially nodal. Needs scaling analysis.")
        b11, b12 = st.columns(2)
        with b11:
            st.download_button(
                "Save Figure",
                data=pdf_cut,
                file_name="band_cut.pdf",
                mime="application/pdf",
                key="download_band_cut_pdf",
                use_container_width=True,
            )
        with b12:
            st.download_button(
                "Save Data",
                data=res["csv_cut"],
                file_name="band_cut.csv",
                mime="text/csv",
                key="download_band_cut_csv",
                use_container_width=True,
            )
    with c2:
        contour_zoom_ctrl = float(st.session_state.get("contour_zoom_ctrl", 0.0))
        contour_zoom = _scale_from_center_slider(contour_zoom_ctrl)
        fig_contour = _plot_band_contour_from_data(
            res["contour_data"],
            int(res["topo_band_index"]),
            float(contour_zoom),
        )
        pdf_contour = _figure_to_pdf_bytes(fig_contour)
        st.pyplot(fig_contour, clear_figure=True)
        st.slider(
            "Band contour color scale",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="contour_zoom_ctrl",
        )
        st.caption(f"Band contour scale = {contour_zoom:.3g}")
        if res["gapless_warning"]:
            st.warning("Potentially nodal. Needs scaling analysis.")
        b21, b22 = st.columns(2)
        with b21:
            st.download_button(
                "Save Figure",
                data=pdf_contour,
                file_name="band_contour.pdf",
                mime="application/pdf",
                key="download_band_contour_pdf",
                use_container_width=True,
            )
        with b22:
            st.download_button(
                "Save Data",
                data=res["csv_contour"],
                file_name=f"band_contour_band_{int(res['topo_band_index'])}.csv",
                mime="text/csv",
                key="download_band_contour_csv",
                use_container_width=True,
            )
    with c3:
        berry_zoom_ctrl = float(st.session_state.get("berry_zoom_ctrl", 0.0))
        berry_zoom = _scale_from_center_slider(berry_zoom_ctrl)

        berry_cache = st.session_state.get("_berry_panel_cache")
        cache_hit = (
            isinstance(berry_cache, dict)
            and int(berry_cache.get("run_id", -1)) == int(res["run_id"])
            and np.isclose(float(berry_cache.get("zoom_ctrl", 999.0)), berry_zoom_ctrl)
        )
        if cache_hit:
            berry_png = berry_cache["png"]
            pdf_berry = berry_cache["pdf"]
        else:
            fig_berry = _plot_berry_curvature_from_data(
                res["berry_data"],
                float(res["chern"]),
                int(res["topo_band_index"]),
                float(berry_zoom),
            )
            berry_png = _figure_to_png_bytes(fig_berry)
            pdf_berry = _figure_to_pdf_bytes(fig_berry)
            plt.close(fig_berry)
            st.session_state["_berry_panel_cache"] = {
                "run_id": int(res["run_id"]),
                "zoom_ctrl": float(berry_zoom_ctrl),
                "png": berry_png,
                "pdf": pdf_berry,
            }

        st.image(berry_png, use_container_width=True)
        st.slider(
            "Berry color scale",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="berry_zoom_ctrl",
        )
        st.caption(f"Berry scale = {berry_zoom:.3g}")
        if res["chern_warning"]:
            st.warning("Don't trust! Chern may be unreliable when Δmin < 1e-2.")
        b31, b32 = st.columns(2)
        with b31:
            st.download_button(
                "Save Figure",
                data=pdf_berry,
                file_name=f"berry_curvature_band_{int(res['topo_band_index'])}.pdf",
                mime="application/pdf",
                key="download_berry_curvature_pdf",
                use_container_width=True,
            )
        with b32:
            st.download_button(
                "Save Data",
                data=res["csv_berry"],
                file_name=f"berry_curvature_band_{int(res['topo_band_index'])}.csv",
                mime="text/csv",
                key="download_berry_curvature_csv",
                use_container_width=True,
            )

st.caption("\u00a9 Shi Feng, TU Munich")

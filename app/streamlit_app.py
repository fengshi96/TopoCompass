import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
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
    fig.colorbar(im, ax=ax, label=r"$\Omega(k)$")
    return fig, {"kx": kx, "ky": ky, "curvature": curv_plot}


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
    + J_3 \sum_{\langle\!\langle\!\langle ij \rangle\!\rangle\!\rangle} \mathbf{S}_i \cdot \mathbf{S}_j \\
    &+ A \sum_i \left(S_i^x S_i^y + S_i^y S_i^x\right)
    - h \sum_i \hat{\mathbf{n}} \cdot \mathbf{S}_i .
    \end{aligned}
    """
)
_last_updated = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {_last_updated}")

st.subheader("Model Parameters")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    spin_s = st.number_input(r"$S$ (Spin)", min_value=0.5, value=0.5, step=0.5, format="%.1f")
with pc2:
    anis_a = st.number_input(r"$A$", value=0.0, format="%.6f")
with pc3:
    j1 = st.number_input(r"$J$", value=float(np.cos(5.0 * np.pi / 4.0)), format="%.6f")
with pc4:
    k_term = st.number_input(r"$K$", value=float(np.sin(5.0 * np.pi / 4.0)), format="%.6f")

pc5, pc6, pc7, pc8 = st.columns(4)
with pc5:
    gamma = st.number_input(r"$\Gamma$", value=-0.50, format="%.6f")
with pc6:
    gamma_p = st.number_input(r"$\Gamma'$", value=-0.0, format="%.6f")
with pc7:
    d_term = st.number_input(r"$D$", value=0.0, format="%.6f")
with pc8:
    j2 = st.number_input(r"$J_2$", value=0.0, format="%.6f")

pc9, _, _, _ = st.columns(4)
with pc9:
    j3 = st.number_input(r"$J_3$", value=0.0, format="%.6f")

st.subheader("Field and Direction")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)
with fcol1:
    dir_x = st.number_input("dir_x", value=1.0, format="%.6f")
with fcol2:
    dir_y = st.number_input("dir_y", value=1.0, format="%.6f")
with fcol3:
    dir_z = st.number_input("dir_z", value=1.0, format="%.6f")
with fcol4:
    b_strength = st.number_input("field_strength", value=4.0, min_value=0.0, format="%.6f")

st.subheader("Resolution")
r1, r2, r3, r4 = st.columns(4)
with r1:
    path_pts = st.number_input("k-path points/segment", min_value=20, max_value=400, value=30, step=10)
with r2:
    contour_nk = st.number_input("Band contour grid", min_value=20, max_value=401, value=30, step=10)
with r3:
    chern_grid = st.number_input("Berry curvature grid", min_value=60, max_value=301, value=60, step=10)
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

if st.button("Run", type="primary"):
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
        with st.spinner("Computing band cut, contour, Berry curvature, and Chern number..."):
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
                int(contour_nk),
                chern,
                topo_band_index,
            )

            pdf_cut = _figure_to_pdf_bytes(fig_cut)
            pdf_contour = _figure_to_pdf_bytes(fig_contour)
            pdf_berry = _figure_to_pdf_bytes(fig_berry)

            cut_header = "s,kx,ky,band0,band1"
            cut_data = np.column_stack([s_vals, k_vals[:, 0], k_vals[:, 1], bands[:, 0], bands[:, 1]])
            csv_cut = _matrix_to_csv_bytes(cut_data, cut_header)
            csv_contour = _grid_to_csv_bytes(contour_data["kx"], contour_data["ky"], contour_data["energy"], "energy")
            csv_berry = _grid_to_csv_bytes(berry_data["kx"], berry_data["ky"], berry_data["curvature"], "curvature")
    except RuntimeError as exc:
        st.error("Solver failed for this parameter set.")
        st.caption(str(exc))
        st.info(
            "Try increasing field_strength, changing exchange parameters, or reducing contour/chern grid to scan a nearby stable region."
        )
        st.stop()

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2], vertical_alignment="top")
    with c1:
        st.pyplot(fig_cut, clear_figure=True)
        if gapless_warning:
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
                data=csv_cut,
                file_name="band_cut.csv",
                mime="text/csv",
                key="download_band_cut_csv",
                use_container_width=True,
            )
    with c2:
        st.pyplot(fig_contour, clear_figure=True)
        if gapless_warning:
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
                data=csv_contour,
                file_name=f"band_contour_band_{topo_band_index}.csv",
                mime="text/csv",
                key="download_band_contour_csv",
                use_container_width=True,
            )
    with c3:
        st.pyplot(fig_berry, clear_figure=True)
        if chern_warning:
            st.warning("Don't trust! Chern may be unreliable when Δmin < 1e-2.")
        b31, b32 = st.columns(2)
        with b31:
            st.download_button(
                "Save Figure",
                data=pdf_berry,
                file_name=f"berry_curvature_band_{topo_band_index}.pdf",
                mime="application/pdf",
                key="download_berry_curvature_pdf",
                use_container_width=True,
            )
        with b32:
            st.download_button(
                "Save Data",
                data=csv_berry,
                file_name=f"berry_curvature_band_{topo_band_index}.csv",
                mime="text/csv",
                key="download_berry_curvature_csv",
                use_container_width=True,
            )

st.caption("\u00a9 Shi Feng, TU Munich")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from io import BytesIO

import sys
from pathlib import Path

# Support running from source layout without requiring editable install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from topocompass import SpinExchangeModel
from topocompass.core import MagnonLSWT

try:
    from topocompass.core_numba import MagnonLSWTNumba as _PreferredSolver

    _SOLVER_LABEL = "Numba-accelerated solver"
except Exception:
    _PreferredSolver = MagnonLSWT
    _SOLVER_LABEL = "Base solver (Numba unavailable)"


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


def _build_kpath(points_per_segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels_to_coords = {
        "K": np.array([2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 0.0]),
        "G": np.array([0.0, 0.0, 0.0]),
        "M": np.array([np.pi, 0.0, 0.0]),
    }
    path = ["K", "G", "M", "K"]
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
    return np.asarray(s_vals), np.asarray(k_vals), s_nodes, path


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


def _plot_band_contour(solver, nk: int):
    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)

    kx = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    ky = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    z = np.empty((nk, nk), dtype=float)
    for i, k1 in enumerate(kx):
        for j, k2 in enumerate(ky):
            q = bmat @ np.array([k1, k2], dtype=float)
            evals, _ = solver.paraunitary_diagonalize(solver.build_magnon_bilinear((q[0], q[1], 0.0)))
            z[j, i] = float(np.sort(evals)[0])

    bz_q = (2.0 * np.pi / 3.0) * np.array(
        [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]], dtype=float
    )
    bz_plot = bz_q @ inv_bmat.T

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
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("Band Contour (Lower)")
    fig.colorbar(im, ax=ax, label=r"$E(k)$")
    return fig


def _plot_berry_curvature(solver, nk: int, chern: float, band_index: int):
    bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
    inv_bmat = np.linalg.inv(bmat)

    payload = solver.derive_berry_curvature(
        grid_n=nk,
        band_index=band_index,
        method="paraunitary",
    )
    curv_q = payload["curvature"]

    kx = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    ky = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk)
    curv_plot = np.empty((nk, nk), dtype=float)
    for i, k1 in enumerate(kx):
        for j, k2 in enumerate(ky):
            q = bmat @ np.array([k1, k2], dtype=float)
            qx = (q[0] % (2.0 * np.pi)) / (2.0 * np.pi) * curv_q.shape[0]
            qy = (q[1] % (2.0 * np.pi)) / (2.0 * np.pi) * curv_q.shape[1]

            # Periodic bilinear remap on the same selected band.
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
        vmin=-10.0,
        vmax=10.0,
        aspect="auto",
    )
    ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.9)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
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
    return fig


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
    H = &\sum_{\langle ij \rangle_{\gamma}} \Big[
    J\,\mathbf{S}_i \cdot \mathbf{S}_j
    + K\,S_i^{\gamma} S_j^{\gamma}
    + \Gamma\,(S_i^{\alpha}S_j^{\beta}+S_i^{\beta}S_j^{\alpha})
    + \Gamma'\,(S_i^{\gamma}S_j^{\alpha}+S_i^{\alpha}S_j^{\gamma}+S_i^{\gamma}S_j^{\beta}+S_i^{\beta}S_j^{\gamma})
    \Big] \\
    &+ J_2 \sum_{\langle\langle ij \rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j
    + J_3 \sum_{\langle\langle\langle ij \rangle\rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j
    + D \sum_{\langle ij \rangle} (\mathbf{S}_i \times \mathbf{S}_j)_z
    + A \sum_i (S_i^z)^2
    - h \sum_i \hat{\mathbf{n}} \cdot \mathbf{S}_i .
    \end{aligned}
    """
)
st.caption(f"Current backend: {_SOLVER_LABEL}.")

st.subheader("Model Parameters")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    spin_s = st.number_input(r"$S$ (Spin)", min_value=0.5, value=0.5, step=0.5, format="%.1f")
with pc2:
    anis_a = st.number_input(r"$A$", value=0.0, format="%.6f")
with pc3:
    j1 = st.number_input(r"$J$", value=0.0, format="%.6f")
with pc4:
    k_term = st.number_input(r"$K$", value=-1.0, format="%.6f")

pc5, pc6, pc7, pc8 = st.columns(4)
with pc5:
    gamma = st.number_input(r"$\Gamma$", value=-0.30, format="%.6f")
with pc6:
    gamma_p = st.number_input(r"$\Gamma'$", value=-0.54, format="%.6f")
with pc7:
    d_term = st.number_input(r"$D$", value=0.0, format="%.6f")
with pc8:
    j2 = st.number_input(r"$J_2$", value=0.0, format="%.6f")

pc9, _, _, _ = st.columns(4)
with pc9:
    j3 = st.number_input(r"$J_3$", value=0.2, format="%.6f")

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
r1, r2, r3 = st.columns(3)
with r1:
    path_pts = st.number_input("k-path points/segment", min_value=20, max_value=400, value=90, step=10)
with r2:
    contour_nk = st.number_input("contour grid", min_value=31, max_value=401, value=121, step=10)
with r3:
    chern_grid = st.number_input("chern grid", min_value=21, max_value=301, value=81, step=10)

band_choice = st.selectbox(
    "Topology band",
    options=["Lower positive band (index 0)", "Upper positive band (index 1)"],
    index=1,
)
topo_band_index = 0 if band_choice.startswith("Lower") else 1

if st.button("Run", type="primary"):
    direction = np.array([dir_x, dir_y, dir_z], dtype=float)
    if np.linalg.norm(direction) < 1e-12:
        st.error("Field direction cannot be zero vector.")
        st.stop()

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
        (float(direction[0]), float(direction[1]), float(direction[2])),
        (float(direction[0]), float(direction[1]), float(direction[2])),
    )

    model = SpinExchangeModel(
        exchanges=pvals,  # type: ignore[arg-type]
        magnetic_field_xyz=(float(direction[0]), float(direction[1]), float(direction[2])),
        symmetry="C3i",
    )
    solver = _PreferredSolver(model)

    with st.spinner("Computing band cut, contour, Berry curvature, and Chern number..."):
        s_vals, k_vals, s_nodes, labels = _build_kpath(int(path_pts))
        bands = solver.solve_band_structure(k_vals)
        fig_cut = _plot_band_cut(s_vals, bands, s_nodes, labels)
        fig_contour = _plot_band_contour(solver, int(contour_nk))
        payload = solver.derive_berry_curvature(
            grid_n=int(chern_grid),
            band_index=topo_band_index,
            method="paraunitary",
        )
        chern = solver.compute_chern_number(payload)
        fig_berry = _plot_berry_curvature(solver, int(contour_nk), chern, topo_band_index)

        pdf_cut = _figure_to_pdf_bytes(fig_cut)
        pdf_contour = _figure_to_pdf_bytes(fig_contour)
        pdf_berry = _figure_to_pdf_bytes(fig_berry)

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2], vertical_alignment="top")
    with c1:
        st.pyplot(fig_cut, clear_figure=True)
        st.download_button(
            "Download",
            data=pdf_cut,
            file_name="band_cut.pdf",
            mime="application/pdf",
            key="download_band_cut_pdf",
            use_container_width=True,
        )
    with c2:
        st.pyplot(fig_contour, clear_figure=True)
        st.download_button(
            "Download",
            data=pdf_contour,
            file_name="band_contour.pdf",
            mime="application/pdf",
            key="download_band_contour_pdf",
            use_container_width=True,
        )
    with c3:
        st.pyplot(fig_berry, clear_figure=True)
        st.download_button(
            "Download",
            data=pdf_berry,
            file_name="berry_curvature_lower.pdf",
            mime="application/pdf",
            key="download_berry_curvature_pdf",
            use_container_width=True,
        )

st.caption("\u00a9 Shi Feng, TU Munich")

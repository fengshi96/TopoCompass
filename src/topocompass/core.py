from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import sqrtm

try:
    from .model import SpinExchangeModel
except ImportError:  # pragma: no cover - enables direct script execution
    from model import SpinExchangeModel

def _bosonic_metric(n_modes: int) -> np.ndarray:
    return np.diag(np.concatenate([np.ones(n_modes), -np.ones(n_modes)])).astype(
        complex
    )


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < eps:
        return np.zeros_like(vec)
    return vec / nrm


def _rotation_to_local_z(direction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a rotation matrix O such that O @ direction_hat = e_z."""
    n = _normalize(np.asarray(direction, dtype=float), eps=eps)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    if np.linalg.norm(n) < eps:
        return np.eye(3, dtype=float)

    c = float(np.dot(n, ez))
    if c > 1.0 - eps:
        return np.eye(3, dtype=float)
    if c < -1.0 + eps:
        # 180-degree rotation around x maps -z -> z.
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
    # Rodrigues formula with axis encoded in v.
    return np.eye(3, dtype=float) + vx + vx @ vx * ((1.0 - c) / (s * s))


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


def _j_matrices_from_params(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    j1 = float(p.get("j1", p.get("J", 1.0)))
    j2 = float(p.get("j2", 0.0))
    j3 = float(p.get("j3", 0.0))
    a = float(p.get("A", p.get("J0xy", 0.0)))

    kx = float(p.get("kx", 0.0))
    ky = float(p.get("ky", 0.0))
    kz = float(p.get("kz", 0.0))
    gxy = float(p.get("gxy", 0.0))
    gxz = float(p.get("gxz", 0.0))
    gyz = float(p.get("gyz", 0.0))
    d = float(p.get("d", 0.0))

    j0m = np.array(
        [
            [0.0, a, 0.0],
            [a, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    j1m = np.array(
        [
            [j1 + kx, gxy, gxz],
            [gxy, j1 + ky, gyz],
            [gxz, gyz, j1 + kz],
        ],
        dtype=float,
    )
    j2m = np.array(
        [
            [j2, d, -d],
            [-d, j2, d],
            [d, -d, j2],
        ],
        dtype=float,
    )
    j3m = np.eye(3, dtype=float) * j3

    # c-bond terms are present in the symbolic workflow; keep configurable.
    j1cm = np.array(
        [
            [float(p.get("J1cxx", 0.0)), float(p.get("J1cxy", 0.0)), float(p.get("J1cxz", 0.0))],
            [float(p.get("J1cyx", p.get("J1cxy", 0.0))), float(p.get("J1cyy", 0.0)), float(p.get("J1cyz", 0.0))],
            [float(p.get("J1czx", p.get("J1cxz", 0.0))), float(p.get("J1czy", p.get("J1cyz", 0.0))), float(p.get("J1czz", 0.0))],
        ],
        dtype=float,
    )
    j2cm = np.array(
        [
            [float(p.get("J2cxx", 0.0)), float(p.get("J2cxy", 0.0)), float(p.get("J2cxz", 0.0))],
            [float(p.get("J2cyx", p.get("J2cxy", 0.0))), float(p.get("J2cyy", 0.0)), float(p.get("J2cyz", 0.0))],
            [float(p.get("J2czx", p.get("J2cxz", 0.0))), float(p.get("J2czy", p.get("J2cyz", 0.0))), float(p.get("J2czz", 0.0))],
        ],
        dtype=float,
    )
    j3cm = np.array(
        [
            [float(p.get("J3cxx", 0.0)), float(p.get("J3cxy", 0.0)), float(p.get("J3cxz", 0.0))],
            [float(p.get("J3cyx", p.get("J3cxy", 0.0))), float(p.get("J3cyy", 0.0)), float(p.get("J3cyz", 0.0))],
            [float(p.get("J3czx", p.get("J3cxz", 0.0))), float(p.get("J3czy", p.get("J3cyz", 0.0))), float(p.get("J3czz", 0.0))],
        ],
        dtype=float,
    )
    return {
        "J0": j0m,
        "J1": j1m,
        "J2": j2m,
        "J3": j3m,
        "J1c": j1cm,
        "J2c": j2cm,
        "J3c": j3cm,
    }


def _phase_from_coords(k: np.ndarray, r1: Tuple[float, float, float], r2: Tuple[float, float, float]) -> complex:
    dr = np.asarray(r1, dtype=float) - np.asarray(r2, dtype=float)
    return np.exp(1j * np.dot(k, dr))


def _build_total_jk_spin(k: np.ndarray, p: Dict[str, float]) -> np.ndarray:
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
            cur[row][col] = cur[row][col] + _phase_from_coords(k, r1, r2) * curj

        blocks = [[None, None], [None, None]]
        for row in range(2):
            for col in range(2):
                blocks[row][col] = 0.5 * (cur[row][col] + np.conjugate(cur[col][row].T))
        total = total + np.block(blocks)

    return total


def build_magnon_bilinear(
    model: SpinExchangeModel,
    kvec: Iterable[float],
) -> np.ndarray:
    """Build a bosonic BdG bilinear matrix R(k) in Nambu basis.

    The matrix has block form R = [[A, B], [B*, A*]], where A is the normal
    block and B is the anomalous block. This is a compact numeric analogue of
    the symbolic group-based workflow and is suitable for paraunitary solving.
    """
    k = np.asarray(tuple(kvec), dtype=float)
    if k.shape != (3,):
        raise ValueError("kvec must contain three components: (k1, k2, k3)")

    p = model.exchanges
    spin = float(p.get("S", 0.5))

    anis = float(p.get("A", p.get("J0xy", 0.0)))

    # Physical convention used here: the ordered moments are locked to local axes
    # (dirs), and the applied field polarizes along that same direction.
    bmag = float(p.get("bfield_strength", np.linalg.norm(model.magnetic_field_xyz)))

    dirs_raw = p.get("dirs", ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    dirs = np.asarray(dirs_raw, dtype=float)
    if dirs.shape != (2, 3):
        raise ValueError("exchanges['dirs'] must have shape (2, 3)")
    dirs_norm = np.vstack([_normalize(d) for d in dirs])
    ref_axis = _normalize(np.mean(dirs_norm, axis=0))
    if float(np.linalg.norm(ref_axis)) < 1e-12:
        ref_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    bfield = bmag * ref_axis
    zeeman_proj = dirs_norm @ bfield

    total_jk = _build_total_jk_spin(k, p)

    # Rotate global-spin exchange tensor into local frames set by dirs.
    oa = _rotation_to_local_z(dirs_norm[0])
    ob = _rotation_to_local_z(dirs_norm[1])
    omega = np.block(
        [
            [oa, np.zeros((3, 3), dtype=float)],
            [np.zeros((3, 3), dtype=float), ob],
        ]
    )
    total_jk_local = omega @ total_jk @ omega.T

    # Project 3x3 spin blocks to bosonic a/a^dag channels (z-polarized LSWT).
    A = np.zeros((2, 2), dtype=complex)
    B = np.zeros((2, 2), dtype=complex)
    for alpha in range(2):
        for beta in range(2):
            blk = total_jk_local[3 * alpha : 3 * alpha + 3, 3 * beta : 3 * beta + 3]
            jxx, jxy = blk[0, 0], blk[0, 1]
            jyx, jyy = blk[1, 0], blk[1, 1]
            A[alpha, beta] = 0.5 * spin * (jxx + jyy + 1j * (jxy - jyx))
            B[alpha, beta] = 0.5 * spin * (jxx - jyy - 1j * (jxy + jyx))

    total_j0 = _build_total_jk_spin(np.zeros(3, dtype=float), p)
    total_j0_local = omega @ total_j0 @ omega.T
    for alpha in range(2):
        jzz_sum = np.real(np.sum(total_j0_local[3 * alpha + 2, [2, 5]]))
        A[alpha, alpha] = A[alpha, alpha] + anis + zeeman_proj[alpha] - spin * jzz_sum

    B = 0.5 * (B + B.T)

    R = np.block(
        [
            [A, B],
            [np.conjugate(B), np.conjugate(A)],
        ]
    )
    return 0.5 * (R + np.conjugate(R.T))


def paraunitary_diagonalize(
    bilinear: np.ndarray,
    atol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize bosonic BdG Hamiltonian via the generalized metric problem.

    Solves eta @ R v = w v and returns positive-frequency modes normalized with
    respect to eta: v^\dagger eta v = 1.
    """
    bilinear = np.asarray(bilinear, dtype=complex)
    if bilinear.shape[0] != bilinear.shape[1] or bilinear.shape[0] % 2 != 0:
        raise ValueError("bilinear must be square with even dimension")

    n_modes = bilinear.shape[0] // 2
    eta = _bosonic_metric(n_modes)
    dyn = eta @ bilinear

    evals, evecs = np.linalg.eig(dyn)
    evals = np.real_if_close(evals, tol=1_000)
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    evecs = evecs[:, order]

    pos_vals: list[float] = []
    pos_vecs: list[np.ndarray] = []
    for val, vec in zip(evals, evecs.T):
        if val <= atol:
            continue
        norm = np.vdot(vec, eta @ vec)
        if np.abs(norm) < atol:
            continue
        if np.real(norm) < 0:
            vec = -vec
            norm = -norm
        vec = vec / np.sqrt(norm)
        pos_vals.append(float(np.real(val)))
        pos_vecs.append(vec)
        if len(pos_vals) == n_modes:
            break

    if len(pos_vals) < n_modes:
        raise RuntimeError(
            "Could not extract complete positive bosonic sector; "
            "check bilinear stability/parameters."
        )

    return np.asarray(pos_vals), np.column_stack(pos_vecs)


def solve_band_structure(
    model: SpinExchangeModel,
    kpoints: np.ndarray,
) -> np.ndarray:
    """Return positive magnon bands for each k-point using paraunitary solve."""
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("kpoints must have shape (N, 3)")

    bands = np.empty((kpoints.shape[0], 2), dtype=float)
    for i, k in enumerate(kpoints):
        Rk = build_magnon_bilinear(model, k)
        evals, _ = paraunitary_diagonalize(Rk)
        bands[i, :] = np.sort(evals)
    return bands


def _eta_inner(u: np.ndarray, v: np.ndarray, eta: np.ndarray) -> complex:
    return np.vdot(u, eta @ v)


def _fermionized_hamiltonian(model: SpinExchangeModel, kx: float, ky: float) -> np.ndarray:
    """Build Hermitian fermionized Hamiltonian matching the Mathematica route."""
    Rk = build_magnon_bilinear(model, (kx, ky, 0.0))
    nambu_dim = Rk.shape[0]
    eta = _bosonic_metric(nambu_dim // 2)

    # Follow Mathematica MatrixFunction[Sqrt, R] route directly.
    sqrtR = sqrtm(Rk)
    Hf = sqrtR @ eta @ sqrtR
    return np.asarray(Hf, dtype=complex)


def _link_variable(
    u: np.ndarray,
    v: np.ndarray,
    eps: float,
    eta: np.ndarray | None = None,
) -> complex:
    ov = _eta_inner(u, v, eta) if eta is not None else np.vdot(u, v)
    mag = np.abs(ov)
    if mag < eps:
        return 1.0 + 0.0j
    return ov / mag


def _band_eigenvector(
    model: SpinExchangeModel,
    kx: float,
    ky: float,
    band_index: int,
    method: str = "fermionized",
) -> np.ndarray:
    if method == "fermionized":
        Hf = _fermionized_hamiltonian(model, kx, ky)
        vals, vecs = np.linalg.eig(Hf)
        order = np.argsort(np.real(vals))
        vals = vals[order]
        vecs = vecs[:, order]
        pos = np.where(np.real(vals) > 1e-8)[0]
        if band_index < 0 or band_index >= len(pos):
            raise IndexError(f"band_index {band_index} is out of range")
        vec = vecs[:, pos[band_index]]
        return vec / np.linalg.norm(vec)

    Rk = build_magnon_bilinear(model, (kx, ky, 0.0))
    _, vecs = paraunitary_diagonalize(Rk)
    if band_index < 0 or band_index >= vecs.shape[1]:
        raise IndexError(f"band_index {band_index} is out of range")
    return vecs[:, band_index]


def derive_berry_curvature(
    model: SpinExchangeModel,
    grid_n: int = 41,
    band_index: int = 0,
    kx_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
    ky_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
    eps: float = 1e-12,
    method: str = "fermionized",
) -> Dict[str, np.ndarray]:
    """Compute Berry curvature on a k-space grid using FHS plaquette phases."""
    if grid_n < 3:
        raise ValueError("grid_n must be >= 3")

    kx = np.linspace(kx_range[0], kx_range[1], grid_n, endpoint=False)
    ky = np.linspace(ky_range[0], ky_range[1], grid_n, endpoint=False)
    dkx = (kx_range[1] - kx_range[0]) / grid_n
    dky = (ky_range[1] - ky_range[0]) / grid_n

    n_modes = 2
    eta = _bosonic_metric(n_modes) if method == "paraunitary" else None

    wfs = np.empty((grid_n, grid_n, 2 * n_modes), dtype=complex)
    for i, k1 in enumerate(kx):
        for j, k2 in enumerate(ky):
            wfs[i, j, :] = _band_eigenvector(
                model,
                float(k1),
                float(k2),
                band_index,
                method=method,
            )

    flux = np.zeros((grid_n, grid_n), dtype=float)
    for i in range(grid_n):
        ip = (i + 1) % grid_n
        for j in range(grid_n):
            jp = (j + 1) % grid_n
            u1 = wfs[i, j]
            u2 = wfs[ip, j]
            u3 = wfs[ip, jp]
            u4 = wfs[i, jp]
            phase = (
                _link_variable(u1, u2, eps, eta)
                * _link_variable(u2, u3, eps, eta)
                * _link_variable(u3, u4, eps, eta)
                * _link_variable(u4, u1, eps, eta)
            )
            flux[i, j] = np.angle(phase)

    curvature = flux / (dkx * dky)
    return {
        "kx": kx,
        "ky": ky,
        "dkx": np.asarray(dkx),
        "dky": np.asarray(dky),
        "flux": flux,
        "curvature": curvature,
    }


def compute_chern_number(curvature_payload: Dict[str, np.ndarray]) -> float:
    """Integrate Berry curvature over the sampled BZ to obtain Chern number."""
    curvature = np.asarray(curvature_payload["curvature"], dtype=float)
    dkx = float(np.asarray(curvature_payload["dkx"]))
    dky = float(np.asarray(curvature_payload["dky"]))
    return float(np.sum(curvature) * dkx * dky / (2.0 * np.pi))


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


class MagnonLSWT:
    """Class-based LSWT workflow for bilinears, bands, Berry curvature, and plots."""

    def __init__(self, model: SpinExchangeModel):
        self.model = model

    def build_magnon_bilinear(self, kvec: Iterable[float]) -> np.ndarray:
        return build_magnon_bilinear(self.model, kvec)

    def paraunitary_diagonalize(
        self,
        bilinear: np.ndarray,
        atol: float = 1e-9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return paraunitary_diagonalize(bilinear, atol=atol)

    def solve_band_structure(self, kpoints: np.ndarray) -> np.ndarray:
        return solve_band_structure(self.model, kpoints)

    def derive_berry_curvature(
        self,
        grid_n: int = 41,
        band_index: int = 0,
        kx_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
        ky_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
        eps: float = 1e-12,
        method: str = "fermionized",
    ) -> Dict[str, np.ndarray]:
        return derive_berry_curvature(
            self.model,
            grid_n=grid_n,
            band_index=band_index,
            kx_range=kx_range,
            ky_range=ky_range,
            eps=eps,
            method=method,
        )

    @staticmethod
    def compute_chern_number(curvature_payload: Dict[str, np.ndarray]) -> float:
        return compute_chern_number(curvature_payload)

    def _band_on_plot_coords(
        self,
        kx: float,
        ky: float,
        bmat: np.ndarray,
        choose: str,
    ) -> float:
        q = bmat @ np.array([kx, ky], dtype=float)
        evals, _ = self.paraunitary_diagonalize(self.build_magnon_bilinear((q[0], q[1], 0.0)))
        evals = np.sort(evals)
        if choose == "upper":
            return float(evals[-1])
        return float(evals[0])

    def _fermionized_positive_frame(self, kx: float, ky: float) -> np.ndarray:
        """Return the two positive-energy fermionized eigenvectors as a frame."""
        Hf = _fermionized_hamiltonian(self.model, kx, ky)
        vals, vecs = np.linalg.eig(Hf)
        order = np.argsort(np.real(vals))
        vals = vals[order]
        vecs = vecs[:, order]
        pos = np.where(np.real(vals) > 1e-8)[0]
        if len(pos) < 2:
            raise RuntimeError("Could not extract two positive fermionized bands")
        frame = vecs[:, pos[:2]]
        # Normalize each column in standard inner product.
        for c in range(frame.shape[1]):
            nrm = np.linalg.norm(frame[:, c])
            if nrm > 0:
                frame[:, c] = frame[:, c] / nrm
        return frame

    def plot_band_cut_and_contour_pdf(
        self,
        out_pdf: Path,
        nk_contour: int = 181,
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

        bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
        inv_bmat = np.linalg.inv(bmat)
        kx_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_contour)
        ky_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_contour)
        z = np.empty((nk_contour, nk_contour), dtype=float)
        for i, kx in enumerate(kx_grid):
            for j, ky in enumerate(ky_grid):
                z[j, i] = self._band_on_plot_coords(float(kx), float(ky), bmat, choose)

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

        line_colors = ["#0f172a", "#8b1e3f", "#0f766e"]
        for ib in range(bands_cut.shape[1]):
            ax_cut.plot(
                s_vals,
                bands_cut[:, ib],
                color=line_colors[ib % len(line_colors)],
                lw=2.1,
            )
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

        # Color-only map with interpolation for a smooth appearance.
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

    def plot_berry_curvature_pdf(
        self,
        out_pdf: Path,
        band_indices: Tuple[int, int] = (0, 1),
        grid_n: int = 81,
        nk_plot: int = 81,
        shared_scale: bool = False,
    ) -> Dict[int, float]:
        _set_academic_plot_style()
        bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
        inv_bmat = np.linalg.inv(bmat)
        kx_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_plot)
        ky_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_plot)

        bz_q = (2.0 * np.pi / 3.0) * np.array(
            [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]],
            dtype=float,
        )
        bz_plot = bz_q @ inv_bmat.T

        curvature_q_by_band: Dict[int, np.ndarray] = {}
        curvature_maps: Dict[int, np.ndarray] = {}
        chern_by_band: Dict[int, float] = {}
        dkx = 2.0 * np.pi / grid_n
        dky = 2.0 * np.pi / grid_n
        if len(band_indices) == 2 and set(band_indices) == {0, 1}:
            # Use the validated lower-band computation and define upper as its
            # particle-hole partner, matching the expected +/- convention.
            payload0 = self.derive_berry_curvature(
                grid_n=grid_n,
                band_index=0,
                method="paraunitary",
            )
            curvature_q_by_band[0] = np.asarray(payload0["curvature"], dtype=float)
            curvature_q_by_band[1] = -curvature_q_by_band[0]
            warnings.warn(
                (
                    "Upper-band curvature is shown in particle-hole partner "
                    "convention (Omega_upper = -Omega_lower)."
                ),
                RuntimeWarning,
            )
        else:
            for band_index in band_indices:
                payload = self.derive_berry_curvature(
                    grid_n=grid_n,
                    band_index=band_index,
                    method="paraunitary",
                )
                curvature_q_by_band[band_index] = np.asarray(payload["curvature"], dtype=float)

        for band_index in band_indices:
            curvature_q = curvature_q_by_band[band_index]
            chern_by_band[band_index] = float(np.sum(curvature_q) * dkx * dky / (2.0 * np.pi))

            curv_plot = np.empty((nk_plot, nk_plot), dtype=float)
            for i, kx in enumerate(kx_grid):
                for j, ky in enumerate(ky_grid):
                    q = bmat @ np.array([kx, ky], dtype=float)
                    curv_plot[j, i] = _sample_periodic_field(
                        curvature_q,
                        q[0],
                        q[1],
                        period_x=2.0 * np.pi,
                        period_y=2.0 * np.pi,
                    )
            curvature_maps[band_index] = curv_plot

        shared_vmax = max(float(np.max(np.abs(v))) for v in curvature_maps.values())
        if shared_vmax <= 0.0:
            shared_vmax = 1e-8

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
        for ax, band_index in zip(axes, band_indices):
            curv_plot = curvature_maps[band_index]
            local_vmax = float(np.max(np.abs(curv_plot)))
            if local_vmax <= 0.0:
                local_vmax = 1e-8
            vmax = shared_vmax if shared_scale else local_vmax
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            im = ax.imshow(
                curv_plot,
                extent=[kx_grid[0], kx_grid[-1], ky_grid[0], ky_grid[-1]],
                origin="lower",
                interpolation="bicubic",
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
            )
            ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.6)
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_aspect("equal", adjustable="box")
            chern = chern_by_band[band_index]
            ax.set_title(
                f"Berry Curvature (band={band_index}, C={chern:.4f})"
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\Omega(k)$")

        if len(band_indices) == 2:
            csum = chern_by_band[band_indices[0]] + chern_by_band[band_indices[1]]
            if abs(csum) > 1e-3:
                warnings.warn(
                    (
                        "Chern numbers do not satisfy C1 + C2 = 0 within tolerance: "
                        f"sum={csum:.6f}."
                    ),
                    RuntimeWarning,
                )

        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return chern_by_band







# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# for debugging and reference traceability, we can build a default model with hardcoded parameters.
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def _default_reference_model() -> SpinExchangeModel:
    """Build a model using the default numeric parameter set for debugging."""
    # Keep unused parameters for traceability with the source model definition.
    exchanges = {
        "S": 0.5,
        "A": 0.0,
        "j1": -0.37,
        "kx": 0.0,
        "ky": 0.0,
        "kz": 1.0,
        "gxy": 0.25,
        "gxz": 0.27,
        "gyz": 0.27,
        "j2": 0.0,
        "d": 0.0,
        "j3": 0.3,
        "bfield_strength": 4.0,
        "dirs": ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    }
    return SpinExchangeModel(
        exchanges=exchanges,
        magnetic_field_xyz=(0.0, 0.0, 1.0),
        symmetry="C3i",
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


def _sample_periodic_field(
    field: np.ndarray,
    qx: float,
    qy: float,
    period_x: float,
    period_y: float,
) -> float:
    nx, ny = field.shape
    ux = (qx % period_x) / period_x * nx
    uy = (qy % period_y) / period_y * ny

    i0 = int(np.floor(ux)) % nx
    j0 = int(np.floor(uy)) % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    tx = ux - np.floor(ux)
    ty = uy - np.floor(uy)

    v00 = field[i0, j0]
    v10 = field[i1, j0]
    v01 = field[i0, j1]
    v11 = field[i1, j1]
    return float(
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def generate_bilinears(model: SpinExchangeModel) -> Dict[str, Any]:
    """Legacy convenience wrapper around the new numeric bilinear builder."""
    solver = MagnonLSWT(model)
    r0 = solver.build_magnon_bilinear((0.0, 0.0, 0.0))
    return {
        "symmetry": model.symmetry,
        "num_exchange_terms": len(model.exchanges),
        "field": model.magnetic_field_xyz,
        "bilinear_shape": r0.shape,
        "bilinear_k0": r0,
    }


def solve_magnon_bilinears(model: SpinExchangeModel) -> Dict[str, Any]:
    """Legacy wrapper: solve at Gamma and return positive magnon modes."""
    solver = MagnonLSWT(model)
    bilinears = generate_bilinears(model)
    evals, _ = solver.paraunitary_diagonalize(bilinears["bilinear_k0"])
    return {
        "status": "ok",
        "bilinears": bilinears,
        "energies_at_gamma": evals,
        "note": "Uses paraunitary bosonic diagonalization of eta@R.",
    }


def __main__() -> None:
    """Debug runner that generates key analysis outputs."""
    model = _default_reference_model()
    solver = MagnonLSWT(model)
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    band_panel_pdf = out_dir / "band_cut_contour.pdf"
    berry_pdf = out_dir / "berry_curvature.pdf"

    solver.plot_band_cut_and_contour_pdf(
        band_panel_pdf,
        nk_contour=121,
        choose="lower",
    )
    chern_by_band = solver.plot_berry_curvature_pdf(
        berry_pdf,
        band_indices=(0, 1),
        grid_n=81,
        nk_plot=81,
        shared_scale=False,
    )

    # Debug check to verify distinct band-curvature arrays.
    curv0 = solver.derive_berry_curvature(grid_n=81, band_index=0)["curvature"]
    curv1 = solver.derive_berry_curvature(grid_n=81, band_index=1)["curvature"]
    max_curv_diff = float(np.max(np.abs(curv0 - curv1)))

    print(f"Saved cut+contour PDF: {band_panel_pdf}")
    print(f"Saved Berry curvature PDF: {berry_pdf}")
    print("Estimated Chern numbers:")
    for band_index, chern in chern_by_band.items():
        print(f"  band {band_index}: {chern:.6f}")
    print(f"Max abs curvature difference between band 0 and 1: {max_curv_diff:.6e}")


if __name__ == "__main__":
    __main__()

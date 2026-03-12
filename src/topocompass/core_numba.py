from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple
import warnings

# Keep OpenMP runtime output quiet and use non-deprecated active-level setting.
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numba import njit, prange

try:
    from .core import MagnonLSWT, _set_academic_plot_style
    from .model import SpinExchangeModel
except ImportError:  # pragma: no cover - enables direct script execution
    from core import MagnonLSWT, _set_academic_plot_style
    from model import SpinExchangeModel


_NUMBA_RUNTIME_OK: bool | None = None


def _try_numba_call(fn, *args):
    global _NUMBA_RUNTIME_OK
    if _NUMBA_RUNTIME_OK is False:
        return None
    try:
        out = fn(*args)
        _NUMBA_RUNTIME_OK = True
        return out
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _NUMBA_RUNTIME_OK = False
        warnings.warn(
            f"Numba JIT unavailable at runtime ({exc}); using Python fallback.",
            RuntimeWarning,
        )
        return None


@njit(cache=True)
def _sample_periodic_field_numba(
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
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def _sample_periodic_field_py(
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
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


@njit(cache=True, parallel=True)
def _map_field_to_plot_grid_numba(
    field_q: np.ndarray,
    bmat: np.ndarray,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    period_x: float,
    period_y: float,
) -> np.ndarray:
    out = np.empty((ky_grid.size, kx_grid.size), dtype=np.float64)
    for i in prange(kx_grid.size):
        kx = kx_grid[i]
        for j in range(ky_grid.size):
            ky = ky_grid[j]
            qx = bmat[0, 0] * kx + bmat[0, 1] * ky
            qy = bmat[1, 0] * kx + bmat[1, 1] * ky
            out[j, i] = _sample_periodic_field_numba(field_q, qx, qy, period_x, period_y)
    return out


def _map_field_to_plot_grid_py(
    field_q: np.ndarray,
    bmat: np.ndarray,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    period_x: float,
    period_y: float,
) -> np.ndarray:
    out = np.empty((ky_grid.size, kx_grid.size), dtype=np.float64)
    for i, kx in enumerate(kx_grid):
        for j, ky in enumerate(ky_grid):
            qx = bmat[0, 0] * kx + bmat[0, 1] * ky
            qy = bmat[1, 0] * kx + bmat[1, 1] * ky
            out[j, i] = _sample_periodic_field_py(field_q, qx, qy, period_x, period_y)
    return out


@njit(cache=True)
def _u1_overlap(ua: np.ndarray, ub: np.ndarray) -> complex:
    ov = 0.0 + 0.0j
    for n in range(ua.size):
        ov += np.conjugate(ua[n]) * ub[n]
    mag = np.abs(ov)
    if mag <= np.finfo(np.float64).tiny:
        return 1.0 + 0.0j
    return ov / mag


@njit(cache=True)
def _u1_overlap_eta(ua: np.ndarray, ub: np.ndarray) -> complex:
    # eta = diag(1,1,-1,-1) for 2-mode bosonic Nambu basis.
    ov = (
        np.conjugate(ua[0]) * ub[0]
        + np.conjugate(ua[1]) * ub[1]
        - np.conjugate(ua[2]) * ub[2]
        - np.conjugate(ua[3]) * ub[3]
    )
    mag = np.abs(ov)
    if mag <= np.finfo(np.float64).tiny:
        return 1.0 + 0.0j
    return ov / mag


def _u1_overlap_py(ua: np.ndarray, ub: np.ndarray) -> complex:
    ov = np.vdot(ua, ub)
    mag = np.abs(ov)
    if mag <= np.finfo(float).tiny:
        return 1.0 + 0.0j
    return ov / mag


def _u1_overlap_eta_py(ua: np.ndarray, ub: np.ndarray) -> complex:
    ov = (
        np.conjugate(ua[0]) * ub[0]
        + np.conjugate(ua[1]) * ub[1]
        - np.conjugate(ua[2]) * ub[2]
        - np.conjugate(ua[3]) * ub[3]
    )
    mag = np.abs(ov)
    if mag <= np.finfo(float).tiny:
        return 1.0 + 0.0j
    return ov / mag


@njit(cache=True, parallel=True)
def _flux_from_wfs_numba(wfs: np.ndarray, use_eta: bool) -> np.ndarray:
    n1, n2, _ = wfs.shape
    flux = np.zeros((n1, n2), dtype=np.float64)

    for i in prange(n1):
        ip = (i + 1) % n1
        for j in range(n2):
            jp = (j + 1) % n2
            u1 = wfs[i, j]
            u2 = wfs[ip, j]
            u3 = wfs[ip, jp]
            u4 = wfs[i, jp]

            if use_eta:
                p = _u1_overlap_eta(u1, u2)
                p *= _u1_overlap_eta(u2, u3)
                p *= _u1_overlap_eta(u3, u4)
                p *= _u1_overlap_eta(u4, u1)
            else:
                p = _u1_overlap(u1, u2)
                p *= _u1_overlap(u2, u3)
                p *= _u1_overlap(u3, u4)
                p *= _u1_overlap(u4, u1)

            flux[i, j] = np.angle(p)
    return flux


def _flux_from_wfs_py(wfs: np.ndarray, use_eta: bool) -> np.ndarray:
    n1, n2, _ = wfs.shape
    flux = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        ip = (i + 1) % n1
        for j in range(n2):
            jp = (j + 1) % n2
            u1 = wfs[i, j]
            u2 = wfs[ip, j]
            u3 = wfs[ip, jp]
            u4 = wfs[i, jp]

            if use_eta:
                p = _u1_overlap_eta_py(u1, u2)
                p *= _u1_overlap_eta_py(u2, u3)
                p *= _u1_overlap_eta_py(u3, u4)
                p *= _u1_overlap_eta_py(u4, u1)
            else:
                p = _u1_overlap_py(u1, u2)
                p *= _u1_overlap_py(u2, u3)
                p *= _u1_overlap_py(u3, u4)
                p *= _u1_overlap_py(u4, u1)

            flux[i, j] = np.angle(p)
    return flux


class MagnonLSWTNumba(MagnonLSWT):
    """Numba-accelerated grid routines for Berry and contour postprocessing.

    Physics construction/diagonalization is inherited from `MagnonLSWT` to keep
    consistency with the validated implementation in core.py.
    """

    def derive_berry_curvature_fast(
        self,
        grid_n: int = 81,
        band_index: int = 0,
        method: str = "paraunitary",
        n_workers: int | None = None,
    ) -> Dict[str, np.ndarray]:
        kx = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False)
        ky = np.linspace(0.0, 2.0 * np.pi, grid_n, endpoint=False)
        dkx = (2.0 * np.pi) / grid_n
        dky = (2.0 * np.pi) / grid_n

        # Reuse parent method for wavefunction generation to preserve behavior.
        # Then accelerate plaquette loop in numba.
        if method != "paraunitary":
            raise ValueError("derive_berry_curvature_fast currently supports method='paraunitary' only")

        wfs = np.empty((grid_n, grid_n, 4), dtype=np.complex128)
        ij_pairs = [(i, j) for i in range(grid_n) for j in range(grid_n)]

        def _task(idx_pair: tuple[int, int]) -> tuple[int, int, np.ndarray]:
            i, j = idx_pair
            Rk = self.build_magnon_bilinear((float(kx[i]), float(ky[j]), 0.0))
            _, vecs = self.paraunitary_diagonalize(Rk)
            return i, j, vecs[:, band_index]

        workers = n_workers or max(1, (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, j, vec in ex.map(_task, ij_pairs):
                wfs[i, j, :] = vec

        flux = _try_numba_call(_flux_from_wfs_numba, wfs, True)
        if flux is None:
            flux = _flux_from_wfs_py(wfs, True)
        curvature = flux / (dkx * dky)
        return {
            "kx": kx,
            "ky": ky,
            "dkx": np.asarray(dkx),
            "dky": np.asarray(dky),
            "flux": flux,
            "curvature": curvature,
        }

    def map_curvature_to_plot_grid_fast(
        self,
        curvature_q: np.ndarray,
        nk_plot: int = 241,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=np.float64)
        kx_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_plot)
        ky_grid = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nk_plot)
        mapped = _try_numba_call(
            _map_field_to_plot_grid_numba,
            np.asarray(curvature_q, dtype=np.float64),
            bmat,
            kx_grid,
            ky_grid,
            2.0 * np.pi,
            2.0 * np.pi,
        )
        if mapped is None:
            mapped = _map_field_to_plot_grid_py(
                np.asarray(curvature_q, dtype=np.float64),
                bmat,
                kx_grid,
                ky_grid,
                2.0 * np.pi,
                2.0 * np.pi,
            )
        return kx_grid, ky_grid, mapped

    def plot_berry_curvature_pdf_fast(
        self,
        out_pdf: Path,
        band_indices: Tuple[int, int] = (0, 1),
        grid_n: int = 81,
        nk_plot: int = 241,
    ) -> Dict[int, float]:
        _set_academic_plot_style()

        bmat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]], dtype=float)
        inv_bmat = np.linalg.inv(bmat)

        bz_q = (2.0 * np.pi / 3.0) * np.array(
            [[1, 1], [2, -1], [1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1]],
            dtype=float,
        )
        bz_plot = bz_q @ inv_bmat.T

        chern_by_band: Dict[int, float] = {}
        maps: Dict[int, np.ndarray] = {}
        kx_grid = None
        ky_grid = None

        for band_index in band_indices:
            payload = self.derive_berry_curvature_fast(
                grid_n=grid_n,
                band_index=band_index,
                method="paraunitary",
            )
            chern_by_band[band_index] = self.compute_chern_number(payload)
            kx_grid, ky_grid, maps[band_index] = self.map_curvature_to_plot_grid_fast(
                payload["curvature"],
                nk_plot=nk_plot,
            )

        shared_vmax = max(float(np.max(np.abs(v))) for v in maps.values())
        shared_vmax = max(shared_vmax, 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
        for ax, band_index in zip(axes, band_indices):
            curv_plot = maps[band_index]
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-shared_vmax, vmax=shared_vmax)
            im = ax.imshow(
                curv_plot,
                extent=[kx_grid[0], kx_grid[-1], ky_grid[0], ky_grid[-1]],
                origin="lower",
                interpolation="bicubic",
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
            )
            ax.plot(bz_plot[:, 0], bz_plot[:, 1], color="#e11d48", ls="--", lw=0.8)
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Berry Curvature (band={band_index}, C={chern_by_band[band_index]:.4f})")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\Omega(k)$")

        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return chern_by_band


def __main__() -> None:
    # This matches the current debug defaults in core.py.
    model = SpinExchangeModel(
        exchanges={
            "S": 0.5,
            "A": 0.0,
            "j1": 0.0,
            "kx": 0.0,
            "ky": 0.0,
            "kz": 1.0,
            "gxy": 0.5,
            "gxz": 0.0,
            "gyz": 0.0,
            "j2": 0.0,
            "d": 0.0,
            "j3": 0.0,
            "bfield_strength": 4.0,
            "dirs": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        },
        magnetic_field_xyz=(1.0, 1.0, 1.0),
        symmetry="C3i",
    )

    solver = MagnonLSWTNumba(model)
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "berry_curvature_numba.pdf"

    chern = solver.plot_berry_curvature_pdf_fast(
        out_pdf=out_pdf,
        band_indices=(0, 1),
        grid_n=81,
        nk_plot=241,
    )
    print(f"Saved Numba Berry PDF: {out_pdf}")
    for b, c in chern.items():
        print(f"  band {b}: C = {c:.6f}")


if __name__ == "__main__":
    __main__()

import sys

sys.path.insert(0, 'src')

from topocompass.model import SpinExchangeModel
from topocompass.core import MagnonLSWT

ex = {
    'A': 0.0,
    'j1': 0.0,
    'kx': 0.0,
    'ky': 0.0,
    'kz': -1.0,
    'gxy': -0.30,
    'gxz': -0.54,
    'gyz': -0.54,
    'j2': 0.0,
    'd': 0.0,
    'j3': 0.2,
    'S': 0.5,
    'bfield_strength': 4.0,
    'dirs': ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}

model = SpinExchangeModel(exchanges=ex, magnetic_field_xyz=(1.0, 1.0, 1.0), symmetry='C3i')
solver = MagnonLSWT(model)

print('paraunitary', [solver.compute_chern_number(solver.derive_berry_curvature(grid_n=81, band_index=b, method='paraunitary')) for b in (0, 1)])
print('fermionized_legacy', [solver.compute_chern_number(solver.derive_berry_curvature(grid_n=81, band_index=b, method='fermionized', fermionized_route='legacy')) for b in (0, 1)])
print('fermionized_notebook', [solver.compute_chern_number(solver.derive_berry_curvature(grid_n=81, band_index=b, method='fermionized', fermionized_route='notebook')) for b in (0, 1)])

legacy = [
    solver.compute_chern_number(
        solver.derive_berry_curvature(
            grid_n=81,
            band_index=b,
            method='fermionized',
            fermionized_route='legacy',
        )
    )
    for b in (0, 1)
]
print('fermionized_legacy_sum_two_positive', sum(legacy))

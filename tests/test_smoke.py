from topocompass import SpinExchangeModel, solve_magnon_bilinears


def test_solver_smoke():
    model = SpinExchangeModel(exchanges={"J": 1.0}, magnetic_field_xyz=(0.0, 0.0, 1.0))
    out = solve_magnon_bilinears(model)
    assert out["status"] == "ok"
    assert out["bilinears"]["symmetry"] == "C3i"

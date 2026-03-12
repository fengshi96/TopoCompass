# TopoCompass

Automated generator and solver of magnon bilinears for generic spin-orbit coupled honeycomb Mott insulators with C3i symmetry-allowed spin exchanges and arbitrary 3D magnetic field polarizations.

## Scope

This repository is structured as a research software project with:

- A core Python package for symbolic/model construction and numerical solving
- A Streamlit interface scaffold for interactive exploration
- Tests and reproducible project metadata

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Project Layout

```text
TopoCompass/
  app/
    streamlit_app.py
  src/
    topocompass/
      __init__.py
      core.py
      model.py
  tests/
    test_smoke.py
  pyproject.toml
  requirements.txt
```

## Development Notes

- The initial implementa- The initial implementa- The initial implementa-ly- The initial implementa- The initial implementa- Tan/symmetry data structures.
- Extend `src/topocompass/core.py` for bilinear generation and solver workflows.

## License

MIT

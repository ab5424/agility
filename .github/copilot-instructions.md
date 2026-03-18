# Copilot Instructions for Agility

## Repository Overview

**Agility** (**A**tomistic **G**rain Boundary and **I**nterface Uti**lity**) is a Python library for pre- and post-processing polycrystalline and grain-boundary structures for use with atomistic simulation codes (e.g. LAMMPS, VASP). It provides top-level processing using established methods to differentiate between bulk and interface regions, enabling extraction of static and dynamic properties.

## Project Structure

```text
agility/          # Main package source
  __init__.py
  analysis.py     # Core GBStructure class and analysis routines
  minimiser.py    # LAMMPS minimisation helpers
  plotting.py     # Plotting utilities (seaborn/matplotlib)
tests/            # Test suite
  unit/           # Unit tests (no optional backends required)
  integration/    # Integration tests (require optional backends)
  files/          # Test fixture files (e.g. aluminium.lmp, STO_polycrystal.lmp)
  conftest.py     # Shared pytest configuration (sets matplotlib backend to Agg)
docs/             # Sphinx documentation source
examples/         # Jupyter notebook examples
```

## Installation

Install the package with optional backend dependencies using `uv` or `pip`:

```bash
# Minimal install
pip install -e .

# With all optional backends and test dependencies
pip install -e ".[tests,ovito,ase,pymatgen]"

# Using uv (preferred in CI)
uv pip install -e ".[tests,ovito,ase,pymatgen]" --system
```

The package requires Python ≥ 3.10. Supported optional backends: `ovito`, `ase`, `pymatgen`.

## Running Tests

```bash
pytest tests/
# With coverage
pytest --cov=agility --cov-report=xml tests/
```

Tests that depend on optional backends are guarded with `pytest.mark.skipif` and `importlib.util.find_spec`. On Linux, `libegl1-mesa-dev` must be installed for ovito to work (`sudo apt install -y libegl1-mesa-dev`).

## Linting and Type Checking

```bash
# Ruff (linter + formatter)
ruff check .
ruff format --check .

# Type checking
ty check
```

Pre-commit hooks (configured in `.pre-commit-config.yaml`) run ruff, ty, codespell, markdownlint, and nbstripout automatically on commit. Run all hooks manually with:

```bash
pre-commit run --all-files
```

## Code Style and Conventions

- **Formatter / linter**: [Ruff](https://docs.astral.sh/ruff/) with `line-length = 100`, targeting Python 3.10+. All Ruff rules are enabled except those explicitly ignored in `pyproject.toml`.
- **Docstrings**: Google-style (`lint.pydocstyle.convention = "google"`).
- **Imports**: Every module must include `from __future__ import annotations` as the first import (enforced by `lint.isort.required-imports`).
- **Type annotations**: All public functions and methods must be fully annotated. Use `TYPE_CHECKING` guards for annotation-only imports.
- **Copyright header**: Source files start with the MIT license header comment block (see existing files for the template).
- **Backends**: New backend integrations are optional dependencies. Guard backend-specific code with `try/except ImportError` or check availability with `importlib.util.find_spec`.

## Testing Conventions

- Tests use `pytest` and `unittest.TestCase`.
- Test files begin with `from __future__ import annotations`.
- Unit tests live under `tests/unit/` and must not require optional backends.
- Integration tests live under `tests/integration/` and are skipped when optional backends are absent.
- Test fixture files live under `tests/files/`.
- Skip tests for optional backends using:

  ```python
  @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
  ```

- Use `numpy.testing.assert_allclose` for floating-point comparisons.
- Place new tests in the appropriate subdirectory under `tests/` matching the module being tested.

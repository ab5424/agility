"""Unit tests for analysis.py using mock objects — no real backends required."""

from __future__ import annotations

import sys
import types
import warnings
from importlib.util import find_spec
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agility.analysis import (
    GBStructure,
    GBStructureTimeseries,
    get_finder,
    invalid_return_type,
    not_implemented,
)

PYTHON_VERSION = sys.version_info
SKIP_OVITO = PYTHON_VERSION <= (3, 12)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _rotation_quat(axis: list[float], angle_deg: float) -> np.ndarray:
    """Return a scalar-last unit quaternion for a rotation by *angle_deg* around *axis*."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    half = np.radians(angle_deg) / 2.0
    return np.array([*(a * np.sin(half)), np.cos(half)])


def _make_gbs(backend: str = "lammps") -> GBStructure:
    """Return a GBStructure with a mocked backend (no real backend packages required)."""
    gbs = GBStructure.__new__(GBStructure)
    gbs.backend = backend
    return gbs


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNotImplemented(TestCase):
    """Test the ``not_implemented`` helper function."""

    def test_returns_not_implemented_error(self) -> None:
        """``not_implemented`` must return a ``NotImplementedError`` instance."""
        err = not_implemented("babel")
        assert isinstance(err, NotImplementedError)

    def test_message_contains_backend_name(self) -> None:
        """The error message must include the backend name."""
        backend = "pyiron"
        err = not_implemented(backend)
        assert backend in str(err)

    def test_can_be_raised(self) -> None:
        """The returned error must be raisable."""
        backend = "babel"
        with pytest.raises(NotImplementedError, match=backend):
            raise not_implemented(backend)


@pytest.mark.unit
class TestInvalidReturnType(TestCase):
    """Test the ``invalid_return_type`` helper function."""

    def test_returns_value_error(self) -> None:
        """``invalid_return_type`` must return a ``ValueError`` instance."""
        err = invalid_return_type("Bogus")
        assert isinstance(err, ValueError)

    def test_message_contains_return_type(self) -> None:
        """The error message must include the invalid return type."""
        err = invalid_return_type("Bogus")
        assert "Bogus" in str(err)

    def test_message_mentions_valid_types(self) -> None:
        """The error message must mention the valid return types."""
        err = invalid_return_type("Bogus")
        assert "Identifier" in str(err)
        assert "Indices" in str(err)

    def test_can_be_raised(self) -> None:
        """The returned error must be raisable."""
        err = invalid_return_type("Bogus")
        with pytest.raises(ValueError, match="Bogus"):
            raise err


@pytest.mark.unit
class TestGetFinder(TestCase):
    """Test the ``get_finder`` helper function (ovito-only, mocked)."""

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_cutoff_finder(self) -> None:
        """``get_finder`` must return a ``CutoffNeighborFinder`` when cutoff is given."""
        with patch("ovito.data.CutoffNeighborFinder") as mock_cutoff:
            mock_cutoff.return_value = "cutoff_finder"
            result = get_finder(MagicMock(), cutoff=3.5)
            assert result == "cutoff_finder"
            mock_cutoff.assert_called_once()

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_nearest_n_finder(self) -> None:
        """``get_finder`` must return a ``NearestNeighborFinder`` when nearest_n is given."""
        with patch("ovito.data.NearestNeighborFinder") as mock_nn:
            mock_nn.return_value = "nn_finder"
            result = get_finder(MagicMock(), nearest_n=12)
            assert result == "nn_finder"
            mock_nn.assert_called_once()

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_no_arguments_raises(self) -> None:
        """``get_finder`` must raise when neither cutoff nor nearest_n is given."""
        with pytest.raises(NameError, match="Either cutoff or nearest_n"):
            get_finder(MagicMock())


# ---------------------------------------------------------------------------
# GBStructure.__init__ and read_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGBStructureInit(TestCase):
    """Test ``GBStructure.__init__`` through the real constructor."""

    def test_init_stores_backend_and_filename(self) -> None:
        """``__init__`` must store the backend and filename attributes."""
        gbs = GBStructure("ase", None)
        assert gbs.backend == "ase"
        assert gbs.filename is None
        assert gbs.data is None

    def test_init_no_filename_skips_read(self) -> None:
        """When ``filename`` is falsy, ``read_file`` must not be called."""
        with patch.object(GBStructure, "read_file") as mock_read:
            GBStructure("ase", None)
            mock_read.assert_not_called()

    def test_init_with_filename_calls_read_file(self) -> None:
        """When ``filename`` is given, ``read_file`` must be called with it."""
        with patch.object(GBStructure, "read_file") as mock_read:
            GBStructure("ase", "dummy.vasp")
            mock_read.assert_called_once_with("dummy.vasp")

    def test_init_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` during ``read_file``."""
        with pytest.raises(NotImplementedError, match="unsupported"):
            GBStructure("unsupported", "dummy.lmp")

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_init_ase_reads_file(self) -> None:
        """Constructing with the ase backend must read the structure from file."""
        with patch("ase.io.read") as mock_read:
            mock_atoms = MagicMock()
            mock_read.return_value = mock_atoms
            gbs = GBStructure("ase", "test.vasp")
            mock_read.assert_called_once_with("test.vasp")
            assert gbs.data.atoms is mock_atoms
            assert gbs.data.selection == []

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_init_ovito_reads_file(self) -> None:
        """Constructing with the ovito backend must create a pipeline."""
        with patch("ovito.io.import_file") as mock_import:
            mock_pipeline = MagicMock()
            mock_import.return_value = mock_pipeline
            gbs = GBStructure("ovito", "test.lmp")
            mock_import.assert_called_once_with("test.lmp")
            assert gbs.pipeline is mock_pipeline

    @pytest.mark.skipif(not find_spec("pymatgen"), reason="pymatgen not installed")
    def test_init_pymatgen_reads_file(self) -> None:
        """Constructing with the pymatgen backend must read the structure from file."""
        with patch("pymatgen.core.Structure") as mock_structure_cls:
            mock_structure = MagicMock()
            mock_structure_cls.from_file.return_value = mock_structure
            gbs = GBStructure("pymatgen", "test.vasp")
            mock_structure_cls.from_file.assert_called_once_with("test.vasp")
            assert gbs.data.structure is mock_structure
            assert gbs.data.selection == []

    def test_init_lammps_creates_pylmp(self) -> None:
        """The lammps backend must create a ``PyLammps`` instance (mocked module)."""
        mock_lammps_module = MagicMock()
        with patch.dict("sys.modules", {"lammps": mock_lammps_module}):
            gbs = GBStructure("lammps", None)
            assert gbs.pylmp is mock_lammps_module.PyLammps.return_value


@pytest.mark.unit
class TestReadFile(TestCase):
    """Test ``GBStructure.read_file`` for each backend (mocked imports)."""

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_read_file_ovito(self) -> None:
        """The ovito backend must call ``import_file`` and store the pipeline."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        with patch("ovito.io.import_file") as mock_import:
            mock_pipeline = MagicMock()
            mock_import.return_value = mock_pipeline
            gbs.read_file("test.lmp")
            mock_import.assert_called_once_with("test.lmp")
            assert gbs.pipeline is mock_pipeline

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_read_file_ase(self) -> None:
        """The ase backend must call ``ase.io.read`` and store atoms + empty selection."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with patch("ase.io.read") as mock_read:
            mock_atoms = MagicMock()
            mock_read.return_value = mock_atoms
            gbs.read_file("test.vasp")
            mock_read.assert_called_once_with("test.vasp")
            assert gbs.data.atoms is mock_atoms
            assert gbs.data.selection == []

    @pytest.mark.skipif(not find_spec("pymatgen"), reason="pymatgen not installed")
    def test_read_file_pymatgen(self) -> None:
        """The pymatgen backend must call ``Structure.from_file`` and store structure."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        gbs.data = None
        with patch("pymatgen.core.Structure") as mock_structure_cls:
            mock_structure = MagicMock()
            mock_structure_cls.from_file.return_value = mock_structure
            gbs.read_file("test.vasp")
            mock_structure_cls.from_file.assert_called_once_with("test.vasp")
            assert gbs.data.structure is mock_structure
            assert gbs.data.selection == []

    def test_read_file_lammps(self) -> None:
        """The lammps backend must delegate to ``_init_lmp``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.data = None
        gbs.pylmp = MagicMock()
        with patch.object(GBStructure, "_init_lmp") as mock_init:
            gbs.read_file("test.data")
            mock_init.assert_called_once_with(filename="test.data")

    def test_read_file_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "babel"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="babel"):
            gbs.read_file("test.xyz")


# ---------------------------------------------------------------------------
# GBStructure._init_lmp
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitLmp(TestCase):
    """Test ``GBStructure._init_lmp`` using a mock LAMMPS object."""

    def setUp(self) -> None:
        """Set up a GBStructure with a mocked pylmp."""
        self.gbs = _make_gbs("lammps")
        self.gbs.pylmp = MagicMock()

    def test_init_lmp_data_file_type(self) -> None:
        """``_init_lmp`` must call ``read_data`` for ``file_type='data'``."""
        self.gbs._init_lmp("test.data", file_type="data")  # noqa: SLF001
        self.gbs.pylmp.read_data.assert_called_once_with("test.data")

    def test_init_lmp_dump_file_type(self) -> None:
        """``_init_lmp`` must call ``read_dump`` for ``file_type='dump'``."""
        self.gbs._init_lmp("test.dump", file_type="dump")  # noqa: SLF001
        self.gbs.pylmp.read_dump.assert_called_once_with("test.dump")

    def test_init_lmp_restart_file_type(self) -> None:
        """``_init_lmp`` must call ``read_restart`` for ``file_type='restart'``."""
        self.gbs._init_lmp("test.restart", file_type="restart")  # noqa: SLF001
        self.gbs.pylmp.read_restart.assert_called_once_with("test.restart")

    def test_init_lmp_invalid_file_type_raises_value_error(self) -> None:
        """``_init_lmp`` must raise ``ValueError`` for an unrecognised file type."""
        with pytest.raises(ValueError, match="type of lammps file"):
            self.gbs._init_lmp("test.xyz", file_type="invalid")  # noqa: SLF001

    def test_init_lmp_sets_units_and_atom_style(self) -> None:
        """``_init_lmp`` must configure units, atom_style, atom_modify, pair_style, kspace_style."""
        self.gbs._init_lmp("test.data")  # noqa: SLF001
        self.gbs.pylmp.units.assert_called_once_with("metal")
        self.gbs.pylmp.atom_style.assert_called_once_with("charge")
        self.gbs.pylmp.atom_modify.assert_called_once_with("map array")
        self.gbs.pylmp.pair_style.assert_called_once_with("none")

    def test_init_lmp_no_kspace_when_none(self) -> None:
        """``_init_lmp`` must not call ``kspace_style`` when ``kspace_style`` is empty."""
        self.gbs._init_lmp("test.data", kspace_style="")  # noqa: SLF001
        self.gbs.pylmp.kspace_style.assert_not_called()

    def test_init_lmp_custom_pair_style(self) -> None:
        """``_init_lmp`` must forward a custom ``pair_style``."""
        self.gbs._init_lmp("test.data", pair_style="lj/cut 10.0")  # noqa: SLF001
        self.gbs.pylmp.pair_style.assert_called_once_with("lj/cut 10.0")


# ---------------------------------------------------------------------------
# GBStructure.save_structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveStructure(TestCase):
    """Test ``GBStructure.save_structure`` for each backend."""

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_save_structure_ase(self) -> None:
        """The ase backend must call ``ase.io.write`` with the correct arguments."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = types.SimpleNamespace(atoms=MagicMock())
        with patch("ase.io.write") as mock_write:
            gbs.save_structure("out.vasp", "vasp")
            mock_write.assert_called_once_with("out.vasp", gbs.data.atoms, format="vasp")

    def test_save_structure_pymatgen(self) -> None:
        """The pymatgen backend must call ``Structure.to`` with the correct arguments."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        mock_structure = MagicMock()
        gbs.data = types.SimpleNamespace(structure=mock_structure)
        gbs.save_structure("out.cif", "cif")
        mock_structure.to.assert_called_once_with(filename="out.cif", fmt="cif")

    def test_save_structure_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "babel"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="babel"):
            gbs.save_structure("out.xyz", "xyz")


@pytest.mark.unit
class TestSaveStructureLammps(TestCase):
    """Test ``save_structure`` for the lammps backend using a mock LAMMPS object."""

    def setUp(self) -> None:
        """Set up a GBStructure with a mocked pylmp."""
        self.gbs = GBStructure.__new__(GBStructure)
        self.gbs.backend = "lammps"
        self.gbs.pylmp = MagicMock()

    def test_save_structure_invalid_file_type_raises_value_error(self) -> None:
        """``save_structure`` must raise ``ValueError`` for an unknown file type."""
        with pytest.raises(ValueError, match="Unrecognised file type"):
            self.gbs.save_structure("out.xyz", "xyz")

    def test_save_structure_data_delegates_to_pylmp(self) -> None:
        """``save_structure`` must call ``write_data`` for ``file_type='data'``."""
        self.gbs.save_structure("out.lmp", "data")
        self.gbs.pylmp.write_data.assert_called_once_with("out.lmp")

    def test_save_structure_dump_delegates_to_pylmp(self) -> None:
        """``save_structure`` must call ``write_dump`` for ``file_type='dump'``."""
        self.gbs.save_structure("out.dump", "dump")
        self.gbs.pylmp.write_dump.assert_called_once_with("out.dump")

    def test_save_structure_restart_delegates_to_pylmp(self) -> None:
        """``save_structure`` must call ``write_restart`` for ``file_type='restart'``."""
        self.gbs.save_structure("out.restart", "restart")
        self.gbs.pylmp.write_restart.assert_called_once_with("out.restart")


# ---------------------------------------------------------------------------
# GBStructure.minimise
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMinimise(TestCase):
    """Test ``GBStructure.minimise`` for each backend."""

    def test_minimise_ovito_raises_not_implemented(self) -> None:
        """The ovito backend must raise ``NotImplementedError`` for ``minimise``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        with pytest.raises(NotImplementedError, match="ovito"):
            gbs.minimise()

    def test_minimise_lammps_delegates_to_minimise_lmp(self) -> None:
        """The lammps backend must delegate to ``minimise_lmp``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        with patch("agility.analysis.minimise_lmp") as mock_min:
            mock_min.return_value = gbs.pylmp
            gbs.minimise()
            mock_min.assert_called_once()

    def test_minimise_pymatgen_calls_relax(self) -> None:
        """The pymatgen backend must call ``Structure.relax`` and update the structure."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        mock_structure = MagicMock()
        mock_relaxed = MagicMock()
        mock_structure.relax.return_value = mock_relaxed
        gbs.data = types.SimpleNamespace(structure=mock_structure)
        gbs.minimise()
        mock_structure.relax.assert_called_once_with()
        assert gbs.data.structure is mock_relaxed

    def test_minimise_pymatgen_unwraps_tuple(self) -> None:
        """The pymatgen backend must unwrap a tuple return from ``relax``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        mock_structure = MagicMock()
        mock_relaxed = MagicMock()
        mock_trajectory = MagicMock()
        mock_structure.relax.return_value = (mock_relaxed, mock_trajectory)
        gbs.data = types.SimpleNamespace(structure=mock_structure)
        gbs.minimise()
        assert gbs.data.structure is mock_relaxed

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_minimise_ase_no_calculator_raises(self) -> None:
        """The ase backend must raise ``ValueError`` when no calculator is set."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        mock_atoms = MagicMock()
        mock_atoms.calc = None
        gbs.data = types.SimpleNamespace(atoms=mock_atoms)
        with pytest.raises(ValueError, match="No ASE calculator"):
            gbs.minimise()

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_minimise_ase_unknown_optimizer_raises(self) -> None:
        """The ase backend must raise ``ValueError`` for an unknown optimizer string."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        mock_atoms = MagicMock()
        mock_atoms.calc = MagicMock()
        gbs.data = types.SimpleNamespace(atoms=mock_atoms)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            gbs.minimise(optimizer="UNKNOWN")

    def test_minimise_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``minimise``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "babel"
        with pytest.raises(NotImplementedError, match="babel"):
            gbs.minimise()


# ---------------------------------------------------------------------------
# GBStructure.delete_particles
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteParticles(TestCase):
    """Test ``GBStructure.delete_particles`` for each backend."""

    def test_delete_particles_ase(self) -> None:
        """The ase backend must remove atoms whose symbol is in ``particle_type``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        atoms_list = [
            MagicMock(symbol="Na"),
            MagicMock(symbol="Cl"),
            MagicMock(symbol="Na"),
        ]
        mock_atoms = MagicMock()
        mock_atoms.__iter__ = lambda *_: iter(atoms_list)
        mock_atoms.__len__ = lambda *_: 3
        mock_atoms.__getitem__ = lambda *args: (
            [atoms_list[i] for i in args[1]] if isinstance(args[1], list) else atoms_list[args[1]]
        )
        gbs.data = types.SimpleNamespace(atoms=mock_atoms, selection=[])
        gbs.delete_particles({"Na"})
        # After deletion, only Cl should remain
        assert len(gbs.data.atoms) == 1

    def test_delete_particles_pymatgen(self) -> None:
        """The pymatgen backend must call ``remove_species`` and reset selection."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        mock_structure = MagicMock()
        gbs.data = types.SimpleNamespace(structure=mock_structure, selection=[1, 2])
        gbs.delete_particles({"Na"})
        mock_structure.remove_species.assert_called_once_with({"Na"})
        assert gbs.data.selection == []

    def test_delete_particles_lammps(self) -> None:
        """The lammps backend must create a group and delete atoms."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.delete_particles({1})
        gbs.pylmp.group.assert_called_once()
        gbs.pylmp.delete_atoms.assert_called_once()

    def test_delete_particles_babel_is_noop(self) -> None:
        """The babel backend must be a no-op (pass) — no exception raised."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "babel"
        gbs.data = None
        # Should not raise
        gbs.delete_particles({"Na"})


# ---------------------------------------------------------------------------
# GBStructure.select_particles_by_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectParticlesByType(TestCase):
    """Test ``GBStructure.select_particles_by_type``."""

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_select_particles_by_type_ovito_appends_modifier(self) -> None:
        """The ovito backend must append modifiers to the pipeline."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs.select_particles_by_type({"Na"})
        assert gbs.pipeline.modifiers.append.call_count >= 1

    def test_select_particles_by_type_unsupported_backend_is_noop(self) -> None:
        """A non-ovito backend must be a no-op for ``select_particles_by_type``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise — the method only handles ovito
        gbs.select_particles_by_type({"Na"})


# ---------------------------------------------------------------------------
# GBStructure.select_particles
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectParticles(TestCase):
    """Test ``GBStructure.select_particles``."""

    def test_select_particles_ase_sets_selection(self) -> None:
        """The ase/pymatgen backend must store the list of IDs in ``data.selection``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = types.SimpleNamespace(selection=[])
        gbs.select_particles([1, 2, 3], invert=False, delete=False)
        assert gbs.data.selection == [1, 2, 3]

    def test_select_particles_pymatgen_overwrites_selection(self) -> None:
        """The pymatgen backend must warn and overwrite when selection is non-empty."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        gbs.data = types.SimpleNamespace(selection=[0, 1])
        with pytest.warns(UserWarning, match="Overwriting selection"):
            gbs.select_particles([5, 6], invert=False, delete=False)
        assert gbs.data.selection == [5, 6]

    def test_select_particles_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a modifier to the pipeline."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        gbs.pipeline = MagicMock()
        gbs.select_particles([1, 2], invert=False, delete=False)
        assert gbs.pipeline.modifiers.append.call_count >= 1


# ---------------------------------------------------------------------------
# GBStructure._invert_selection / _delete_selection / _clear_selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClearSelectionUnit(TestCase):
    """Test ``_clear_selection`` using mock data objects — no backend packages required."""

    def _make_gbs(self, backend: str, selection: list[int]) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        gbs.data = types.SimpleNamespace(selection=list(selection))
        return gbs

    def test_clear_selection_pymatgen(self) -> None:
        """``_clear_selection`` must reset the selection list for the pymatgen backend."""
        gbs = self._make_gbs("pymatgen", [0, 1, 2])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_selection_ase(self) -> None:
        """``_clear_selection`` must reset the selection list for the ase backend."""
        gbs = self._make_gbs("ase", [3, 5, 7])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_already_empty_selection(self) -> None:
        """``_clear_selection`` must be a no-op when the selection is already empty."""
        gbs = self._make_gbs("pymatgen", [])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_clear_selection_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a ``ClearSelectionModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.pipeline.modifiers.append.call_count >= 1


@pytest.mark.unit
class TestInvertSelectionUnit(TestCase):
    """Test ``_invert_selection`` using mock data objects — no backend packages required."""

    def _make_gbs(self, backend: str, selection: list[int], n_items: int) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        gbs.data = types.SimpleNamespace(selection=list(selection))
        if backend == "pymatgen":
            gbs.data.structure = [None] * n_items
        else:
            gbs.data.atoms = [None] * n_items
        return gbs

    def test_invert_empty_selection_pymatgen(self) -> None:
        """Inverting an empty selection must select all sites (pymatgen backend)."""
        gbs = self._make_gbs("pymatgen", [], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [0, 1, 2, 3]

    def test_invert_full_selection_ase(self) -> None:
        """Inverting a full selection must yield an empty selection (ase backend)."""
        gbs = self._make_gbs("ase", [0, 1, 2, 3], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_invert_partial_selection(self) -> None:
        """Inverting a partial selection must return the complementary set."""
        gbs = self._make_gbs("pymatgen", [0, 2], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [1, 3]

    def test_invert_twice_restores_original(self) -> None:
        """Inverting a selection twice must restore the original selection."""
        original = [1, 3]
        gbs = self._make_gbs("ase", original, 4)
        gbs._invert_selection()  # noqa: SLF001
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == original

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_invert_selection_ovito_appends_modifier(self) -> None:
        """The ovito backend must append an ``InvertSelectionModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.pipeline.modifiers.append.call_count >= 1


@pytest.mark.unit
class TestDeleteSelectionUnit(TestCase):
    """Test ``_delete_selection`` using mock data objects — no backend packages required."""

    def test_delete_selection_pymatgen(self) -> None:
        """The pymatgen backend must call ``remove_sites`` and reset selection."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        mock_structure = MagicMock()
        gbs.data = types.SimpleNamespace(structure=mock_structure, selection=[0, 1])
        gbs._delete_selection()  # noqa: SLF001
        mock_structure.remove_sites.assert_called_once_with([0, 1])
        assert gbs.data.selection == []

    def test_delete_selection_ase(self) -> None:
        """The ase backend must remove selected atoms and reset selection."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        atoms_list = [
            MagicMock(symbol="Na"),
            MagicMock(symbol="Cl"),
            MagicMock(symbol="Na"),
            MagicMock(symbol="Cl"),
        ]
        mock_atoms = MagicMock()
        mock_atoms.__len__ = lambda *_: 4
        mock_atoms.__getitem__ = lambda *args: (
            [atoms_list[i] for i in args[1]] if isinstance(args[1], list) else atoms_list[args[1]]
        )
        gbs.data = types.SimpleNamespace(atoms=mock_atoms, selection=[0, 2])
        gbs._delete_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_delete_selection_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a ``DeleteSelectedModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs._delete_selection()  # noqa: SLF001
        assert gbs.pipeline.modifiers.append.call_count >= 1


# ---------------------------------------------------------------------------
# GBStructure.perform_cna
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformCnaValidation(TestCase):
    """Unit tests for ``perform_cna`` enabled-structure validation — no backend required."""

    def _make_gbs(self, backend: str = "lammps") -> GBStructure:
        """Return a GBStructure with a mocked backend."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        return gbs

    def test_invalid_structure_type_raises_value_error(self) -> None:
        """``perform_cna`` must raise ``ValueError`` for an unknown structure type."""
        gbs = self._make_gbs()
        with pytest.raises(ValueError, match="unknown"):
            gbs.perform_cna(enabled=("fcc", "diamond"), compute=False)

    def test_all_valid_structure_types_accepted(self) -> None:
        """``perform_cna`` must accept all valid structure types without error."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        # Passing all four types suppresses the lammps warning
        gbs.perform_cna(enabled=("fcc", "hcp", "bcc", "ico"), compute=False)

    def test_string_enabled_is_normalised(self) -> None:
        """A plain string passed as ``enabled`` must be treated as a single-element list."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        # Should not raise even though a bare string is passed (warning expected for lammps)
        with pytest.warns(UserWarning, match="lammps"):
            gbs.perform_cna(enabled="fcc", compute=False)

    def test_invalid_string_enabled_raises(self) -> None:
        """A plain invalid string passed as ``enabled`` must raise ``ValueError``."""
        gbs = self._make_gbs()
        with pytest.raises(ValueError, match="unknown"):
            gbs.perform_cna(enabled="diamond", compute=False)

    def test_lammps_warns_when_enabled_not_full_set(self) -> None:
        """``perform_cna`` must warn on the lammps backend when ``enabled`` restricts types."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        with pytest.warns(UserWarning, match="lammps"):
            gbs.perform_cna(enabled=("fcc",), compute=False)

    def test_lammps_no_warning_when_all_types_enabled(self) -> None:
        """``perform_cna`` must not warn on lammps when all four types are enabled."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # No warning should be raised when the full set is passed
            gbs.perform_cna(enabled=("fcc", "hcp", "bcc", "ico"), compute=False)

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_cna_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a ``CommonNeighborAnalysisModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs.perform_cna(enabled=("fcc",), compute=False)
        assert gbs.pipeline.modifiers.append.call_count >= 1

    def test_cna_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``perform_cna``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.perform_cna(enabled=("fcc",), compute=False)


# ---------------------------------------------------------------------------
# GBStructure.perform_cnp
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformCnp(TestCase):
    """Test ``GBStructure.perform_cnp``."""

    def test_perform_cnp_lammps(self) -> None:
        """The lammps backend must call ``compute`` with ``cnp/atom``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.perform_cnp(cutoff=3.2, compute=False)
        gbs.pylmp.compute.assert_called_once_with("compute 1 all cnp/atom 3.2")

    def test_perform_cnp_unsupported_backend_is_noop(self) -> None:
        """A non-lammps backend must be a no-op for ``perform_cnp``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        # Should not raise — the method only handles lammps
        gbs.perform_cnp(compute=False)


# ---------------------------------------------------------------------------
# GBStructure.perform_voronoi_analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformVoronoiAnalysis(TestCase):
    """Test ``GBStructure.perform_voronoi_analysis``."""

    def test_perform_voronoi_lammps(self) -> None:
        """The lammps backend must call ``compute`` with ``voronoi/atom``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.perform_voronoi_analysis(compute=False)
        gbs.pylmp.compute.assert_called_once_with("1 all voronoi/atom")

    def test_perform_voronoi_unsupported_backend_is_noop(self) -> None:
        """A non-ovito/non-lammps backend must be a no-op for ``perform_voronoi_analysis``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise — the method only handles ovito and lammps
        gbs.perform_voronoi_analysis(compute=False)


# ---------------------------------------------------------------------------
# GBStructure.perform_ptm
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformPtmValidation(TestCase):
    """Unit tests for ``perform_ptm`` enabled-structure validation — no backend required."""

    def _make_gbs(self, backend: str = "lammps") -> GBStructure:
        """Return a GBStructure with a mocked backend."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        return gbs

    def test_invalid_structure_type_raises_value_error(self) -> None:
        """``perform_ptm`` must raise ``ValueError`` for an unknown structure type."""
        gbs = self._make_gbs()
        with pytest.raises(ValueError, match="unknown"):
            gbs.perform_ptm(enabled=("fcc", "diamond"), compute=False)

    def test_string_enabled_is_normalised(self) -> None:
        """A plain string passed as ``enabled`` must be treated as a single-element list."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        gbs.perform_ptm(enabled="fcc", compute=False)

    def test_invalid_string_enabled_raises(self) -> None:
        """A plain invalid string passed as ``enabled`` must raise ``ValueError``."""
        gbs = self._make_gbs()
        with pytest.raises(ValueError, match="unknown"):
            gbs.perform_ptm(enabled="diamond", compute=False)

    def test_ptm_lammps_calls_compute(self) -> None:
        """The lammps backend must call ``compute`` with ``ptm/atom``."""
        gbs = self._make_gbs()
        gbs.pylmp = MagicMock()
        gbs.perform_ptm(enabled=("fcc",), compute=False)
        gbs.pylmp.compute.assert_called_once()

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_ptm_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a ``PolyhedralTemplateMatchingModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs.perform_ptm(enabled=("fcc",), compute=False)
        assert gbs.pipeline.modifiers.append.call_count >= 1

    def test_ptm_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``perform_ptm``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.perform_ptm(enabled=("fcc",), compute=False)


# ---------------------------------------------------------------------------
# GBStructure.perform_ajm
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformAjm(TestCase):
    """Test ``GBStructure.perform_ajm``."""

    def test_perform_ajm_lammps(self) -> None:
        """The lammps backend must call ``compute`` with ``ackland/atom``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.perform_ajm(compute=False)
        gbs.pylmp.compute.assert_called_once_with("ackland_0 all ackland/atom")

    def test_perform_ajm_ovito_appends_modifier(self) -> None:
        """The ovito backend must append an ``AcklandJonesModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        mock_ajm = MagicMock()
        mock_module = MagicMock(AcklandJonesModifier=MagicMock(return_value=mock_ajm))
        with patch.dict("sys.modules", {"ovito.plugins.ParticlesPython": mock_module}):
            gbs.perform_ajm(compute=False)
            assert gbs.pipeline.modifiers.append.call_count >= 1

    def test_perform_ajm_unsupported_backend_is_noop(self) -> None:
        """An unsupported backend must be a no-op (pass) for ``perform_ajm``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise
        gbs.perform_ajm(compute=False)


# ---------------------------------------------------------------------------
# GBStructure.perform_csp
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformCsp(TestCase):
    """Test ``GBStructure.perform_csp``."""

    def test_perform_csp_lammps(self) -> None:
        """The lammps backend must call ``compute`` with ``centro/atom``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.perform_csp(num_neighbors=12, compute=False)
        gbs.pylmp.compute.assert_called_once_with("centro_0 all centro/atom 12")

    def test_perform_csp_ovito_appends_modifier(self) -> None:
        """The ovito backend must append a ``CentroSymmetryModifier``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        mock_csp = MagicMock()
        mock_module = MagicMock(CentroSymmetryModifier=MagicMock(return_value=mock_csp))
        with patch.dict("sys.modules", {"ovito.plugins.ParticlesPython": mock_module}):
            gbs.perform_csp(compute=False)
            assert gbs.pipeline.modifiers.append.call_count >= 1

    def test_perform_csp_unsupported_backend_is_noop(self) -> None:
        """A non-ovito/non-lammps backend must be a no-op for ``perform_csp``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise — the method only handles ovito and lammps
        gbs.perform_csp(compute=False)


# ---------------------------------------------------------------------------
# GBStructure.get_distinct_grains
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetDistinctGrains(TestCase):
    """Test ``GBStructure.get_distinct_grains``."""

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_get_distinct_grains_ovito_returns_orientations(self) -> None:
        """The ovito backend must return an ``(N, 4)`` quaternion array when ``compute=True``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        mock_data = MagicMock()
        mock_data.tables = {"grains": MagicMock()}
        mock_data.tables["grains"].__getitem__ = lambda *args: np.array(  # noqa: ARG005
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )
        gbs.pipeline.compute.return_value = mock_data
        orientations = gbs.get_distinct_grains(compute=True)
        assert orientations is not None
        assert orientations.shape == (2, 4)

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_get_distinct_grains_ovito_compute_false_returns_none(self) -> None:
        """The ovito backend must return ``None`` when ``compute=False``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        result = gbs.get_distinct_grains(compute=False)
        assert result is None

    @pytest.mark.skipif(not find_spec("ovito"), reason="ovito not installed")
    @pytest.mark.skipif(SKIP_OVITO, reason="Python <= 3.12 not supported for ovito")
    def test_get_distinct_grains_invalid_algorithm_raises(self) -> None:
        """An invalid algorithm name must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        with pytest.raises(ValueError, match="Incorrect"):
            gbs.get_distinct_grains(algorithm="InvalidAlgorithm", compute=False)

    def test_get_distinct_grains_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``get_distinct_grains``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_distinct_grains()


# ---------------------------------------------------------------------------
# GBStructure.set_analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetAnalysis(TestCase):
    """Test ``GBStructure.set_analysis``."""

    def test_set_analysis_ovito(self) -> None:
        """The ovito backend must call ``pipeline.compute()``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.pipeline = MagicMock()
        gbs.set_analysis()
        gbs.pipeline.compute.assert_called_once_with()

    def test_set_analysis_lammps(self) -> None:
        """The lammps backend must call ``pylmp.run(1)``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.set_analysis()
        gbs.pylmp.run.assert_called_once_with(1)

    def test_set_analysis_pymatgen_raises(self) -> None:
        """The pymatgen backend must raise ``NotImplementedError`` for ``set_analysis``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "pymatgen"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="pymatgen"):
            gbs.set_analysis()

    def test_set_analysis_unsupported_backend_is_noop(self) -> None:
        """A non-ovito/non-lammps/non-pymatgen backend must be a no-op for ``set_analysis``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise — the method only handles ovito, lammps, and pymatgen
        gbs.set_analysis()


# ---------------------------------------------------------------------------
# GBStructure.expand_to_non_selected
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpandToNonSelected(TestCase):
    """Test ``GBStructure.expand_to_non_selected``."""

    def test_expand_to_non_selected_both_nearest_n_and_cutoff_raises(self) -> None:
        """Specifying both ``nearest_n`` and ``cutoff`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        with pytest.raises(ValueError, match="Only one"):
            gbs.expand_to_non_selected(nearest_n=12, cutoff=3.5)

    def test_expand_to_non_selected_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.expand_to_non_selected(nearest_n=12)

    def test_expand_to_non_selected_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.expand_to_non_selected(nearest_n=12, return_type="Bogus")


# ---------------------------------------------------------------------------
# GBStructure.expand_to_non_selected_groups
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpandToNonSelectedGroups(TestCase):
    """Test ``GBStructure.expand_to_non_selected_groups``."""

    def test_expand_to_non_selected_groups_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.expand_to_non_selected_groups(groups=[[0, 1]])

    def test_expand_to_non_selected_groups_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.expand_to_non_selected_groups(groups=[[0, 1]], return_type="Bogus")


# ---------------------------------------------------------------------------
# GBStructure._extract_lammps_structure_ids_and_types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractLammpsStructureIdsAndTypes(TestCase):
    """Test ``GBStructure._extract_lammps_structure_ids_and_types``."""

    def _make_gbs(self) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        return gbs

    def _patch_lammps_imports(self) -> patch:
        """Patch the ``lammps`` module so it can be imported in unit tests."""
        mock_lammps = MagicMock()
        return patch.dict("sys.modules", {"lammps": mock_lammps})

    def test_invalid_mode_raises_value_error(self) -> None:
        """An unrecognised mode must raise ``ValueError``."""
        gbs = self._make_gbs()
        with (
            self._patch_lammps_imports(),
            pytest.raises(ValueError, match="Incorrect mode"),
        ):
            gbs._extract_lammps_structure_ids_and_types("invalid")  # noqa: SLF001

    def test_voronoi_mode_raises_not_implemented(self) -> None:
        """The ``voronoi`` mode must raise ``NotImplementedError``."""
        gbs = self._make_gbs()
        with (
            self._patch_lammps_imports(),
            pytest.raises(NotImplementedError, match="not implemented"),
        ):
            gbs._extract_lammps_structure_ids_and_types("voronoi")  # noqa: SLF001

    def test_centro_mode_raises_not_implemented(self) -> None:
        """The ``centro`` mode must raise ``NotImplementedError``."""
        gbs = self._make_gbs()
        with (
            self._patch_lammps_imports(),
            pytest.raises(NotImplementedError, match="not implemented"),
        ):
            gbs._extract_lammps_structure_ids_and_types("centro")  # noqa: SLF001

    def test_cna_mode_returns_correct_sentinel(self) -> None:
        """The ``cna`` mode must use sentinel value 5 for non-crystalline atoms."""
        gbs = self._make_gbs()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array([1, 2, 5, 0])
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with self._patch_lammps_imports():
            ids, types, sentinel = gbs._extract_lammps_structure_ids_and_types("cna")  # noqa: SLF001
        assert sentinel == 5
        assert len(ids) == 4
        assert len(types) == 4

    def test_ptm_mode_returns_correct_sentinel(self) -> None:
        """The ``ptm`` mode must use sentinel value 0 for non-crystalline atoms."""
        gbs = self._make_gbs()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array(
            [[1], [2], [0], [3]],
        )
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with self._patch_lammps_imports():
            ids, types, sentinel = gbs._extract_lammps_structure_ids_and_types("ptm")  # noqa: SLF001
        assert sentinel == 0
        assert len(ids) == 4
        assert len(types) == 4

    def test_ackland_mode_returns_correct_sentinel(self) -> None:
        """The ``ackland`` mode must use sentinel value 0 for non-crystalline atoms."""
        gbs = self._make_gbs()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array([1, 2, 0, 3])
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with self._patch_lammps_imports():
            ids, types, sentinel = gbs._extract_lammps_structure_ids_and_types("ackland")  # noqa: SLF001
        assert sentinel == 0
        assert len(ids) == 4
        assert len(types) == 4


# ---------------------------------------------------------------------------
# GBStructure.get_non_crystalline_atoms
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNonCrystallineAtoms(TestCase):
    """Test ``GBStructure.get_non_crystalline_atoms``."""

    def test_get_non_crystalline_atoms_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_non_crystalline_atoms()

    def test_get_non_crystalline_atoms_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array([1, 2, 5, 0])
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.get_non_crystalline_atoms(mode="cna", return_type="Bogus")

    def test_get_non_crystalline_atoms_ovito_no_structure_type_warns(self) -> None:
        """The ovito backend must warn when no structure type info is available."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        mock_data = MagicMock()
        # Simulate that neither "Structure Type" nor "Centrosymmetry" is in particles
        mock_data.particles.__contains__ = MagicMock(return_value=False)
        gbs.data = mock_data
        with pytest.raises(NotImplementedError, match="ovito"):
            gbs.get_non_crystalline_atoms()


# ---------------------------------------------------------------------------
# GBStructure.get_crystalline_atoms
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCrystallineAtoms(TestCase):
    """Test ``GBStructure.get_crystalline_atoms``."""

    def test_get_crystalline_atoms_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_crystalline_atoms()

    def test_get_crystalline_atoms_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array([1, 2, 5, 0])
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.get_crystalline_atoms(mode="cna", return_type="Bogus")

    def test_get_crystalline_atoms_ovito_no_structure_type_warns(self) -> None:
        """The ovito backend must warn and return empty list when no structure type info."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        mock_data = MagicMock()
        # Simulate that "Structure Type" is not in particles
        mock_data.particles.__contains__ = MagicMock(return_value=False)
        gbs.data = mock_data
        with pytest.warns(UserWarning, match="No structure type"):
            result = gbs.get_crystalline_atoms()
        assert result == []


# ---------------------------------------------------------------------------
# GBStructure.get_grain_edge_ions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetGrainEdgeIons(TestCase):
    """Test ``GBStructure.get_grain_edge_ions``."""

    def test_get_grain_edge_ions_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_grain_edge_ions()

    def test_get_grain_edge_ions_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.lmp.numpy.extract_compute.return_value = np.array([1, 2, 5, 0])
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.get_grain_edge_ions(
                gb_ions={0},
                bulk_ions=[1, 2, 3],
                return_type="Bogus",
            )


# ---------------------------------------------------------------------------
# GBStructure.get_gb_fraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetGbFraction(TestCase):
    """Test ``GBStructure.get_gb_fraction``."""

    def test_get_gb_fraction_lammps(self) -> None:
        """The lammps backend must return the fraction of non-crystalline atoms."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.system.natoms = 10
        with patch.object(
            GBStructure,
            "get_non_crystalline_atoms",
            return_value=[1, 2, 3],
        ):
            assert gbs.get_gb_fraction() == pytest.approx(0.3)

    def test_get_gb_fraction_lammps_mode_passed_through(self) -> None:
        """The lammps backend must pass the mode to get_non_crystalline_atoms."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.system.natoms = 4
        with patch.object(
            GBStructure,
            "get_non_crystalline_atoms",
            return_value=[1],
        ) as mock_gnc:
            gbs.get_gb_fraction(mode="ptm")
        mock_gnc.assert_called_once_with("ptm")

    def test_get_gb_fraction_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``get_gb_fraction``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_gb_fraction()


# ---------------------------------------------------------------------------
# GBStructure.get_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetType(TestCase):
    """Test ``GBStructure.get_type``."""

    def test_get_type_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``get_type``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_type(1)

    def test_get_type_invalid_return_type_raises(self) -> None:
        """An invalid ``return_type`` must raise ``ValueError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.pylmp.lmp.numpy.extract_atom.return_value = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Invalid return type"):
            gbs.get_type(1, return_type="Bogus")


# ---------------------------------------------------------------------------
# GBStructure.get_tilt_angle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetTiltAngle(TestCase):
    """Test ``GBStructure.get_tilt_angle`` using known quaternion/boundary-normal pairs."""

    def _make_gbs(self, backend: str = "ovito") -> GBStructure:
        """Return a GBStructure with mocked internals (no real backend needed)."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        return gbs

    def test_pure_tilt_returns_correct_angles(self) -> None:
        """30° rotation with axis in boundary plane must give tilt=30°, twist=0°."""
        gbs = self._make_gbs()
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([1, 0, 0], 30.0)])
        tilt, twist = gbs.get_tilt_angle(q_i, q_j, boundary_normal=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(tilt, [30.0], atol=1e-10)
        np.testing.assert_allclose(twist, [0.0], atol=1e-10)

    def test_pure_twist_returns_correct_angles(self) -> None:
        """30° rotation with axis along boundary normal must give tilt=0°, twist=30°."""
        gbs = self._make_gbs()
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([0, 0, 1], 30.0)])
        tilt, twist = gbs.get_tilt_angle(q_i, q_j, boundary_normal=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(tilt, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist, [30.0], atol=1e-10)

    def test_identity_misorientation_returns_zero_angles(self) -> None:
        """q_i == q_j must give tilt=0° and twist=0°."""
        gbs = self._make_gbs()
        q = np.array([[0.0, 0.0, 0.0, 1.0]])
        tilt, twist = gbs.get_tilt_angle(q, q, boundary_normal=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(tilt, [0.0], atol=1e-10)
        np.testing.assert_allclose(twist, [0.0], atol=1e-10)

    def test_returns_numpy_arrays(self) -> None:
        """``get_tilt_angle`` must return numpy arrays."""
        gbs = self._make_gbs()
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([1, 0, 0], 45.0)])
        tilt, twist = gbs.get_tilt_angle(q_i, q_j, boundary_normal=[0.0, 0.0, 1.0])
        assert isinstance(tilt, np.ndarray)
        assert isinstance(twist, np.ndarray)

    def test_angles_are_in_degrees(self) -> None:
        """Returned angles must be in degrees (not radians)."""
        gbs = self._make_gbs()
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([1, 0, 0], 90.0)])
        tilt, _ = gbs.get_tilt_angle(q_i, q_j, boundary_normal=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(tilt, [90.0], atol=1e-10)

    def test_optional_cubic_symmetry_reduction(self) -> None:
        """``get_tilt_angle`` should expose optional internal cubic symmetry reduction."""
        gbs = self._make_gbs()
        q_i = np.array([[0.0, 0.0, 0.0, 1.0]])
        q_j = np.array([_rotation_quat([0, 0, 1], 90.0)])
        _, twist_raw = gbs.get_tilt_angle(
            q_i,
            q_j,
            boundary_normal=[0.0, 0.0, 1.0],
            reduce_cubic_symmetry=False,
        )
        _, twist_red = gbs.get_tilt_angle(
            q_i,
            q_j,
            boundary_normal=[0.0, 0.0, 1.0],
            reduce_cubic_symmetry=True,
        )
        np.testing.assert_allclose(twist_raw, [90.0], atol=1e-10)
        np.testing.assert_allclose(twist_red, [0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# GBStructure.get_fraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetFraction(TestCase):
    """Test ``GBStructure.get_fraction``."""

    def test_get_fraction_unsupported_backend_raises(self) -> None:
        """An unsupported backend must raise ``NotImplementedError`` for ``get_fraction``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.get_fraction([1], [2])


# ---------------------------------------------------------------------------
# GBStructure.save_image
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveImage(TestCase):
    """Test ``GBStructure.save_image``."""

    def test_save_image_ovito_is_noop(self) -> None:
        """The ovito backend must be a no-op (pass) for ``save_image``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ovito"
        gbs.data = None
        # Should not raise
        gbs.save_image("image.png")

    def test_save_image_lammps_calls_image(self) -> None:
        """The lammps backend must call ``pylmp.image``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        gbs.save_image("image.png")
        gbs.pylmp.image.assert_called_once_with(filename="image.png")

    def test_save_image_unsupported_backend_is_noop(self) -> None:
        """An unsupported backend must be a no-op for ``save_image``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        # Should not raise
        gbs.save_image("image.png")


# ---------------------------------------------------------------------------
# GBStructure.convert_backend
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConvertBackend(TestCase):
    """Test ``GBStructure.convert_backend``."""

    def test_convert_backend_non_lammps_raises(self) -> None:
        """``convert_backend`` must raise ``NotImplementedError`` for non-lammps backends."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "ase"
        gbs.data = None
        with pytest.raises(NotImplementedError, match="ase"):
            gbs.convert_backend("ovito")

    def test_convert_backend_lammps_to_unsupported_raises(self) -> None:
        """Converting from lammps to an unsupported backend must raise ``NotImplementedError``."""
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = "lammps"
        gbs.pylmp = MagicMock()
        with pytest.raises(NotImplementedError, match="pymatgen"):
            gbs.convert_backend("pymatgen")


# ---------------------------------------------------------------------------
# GBStructureTimeseries — inheritance, init, num_frames, get_frame, remove_timesteps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGBStructureTimeseriesInheritance(TestCase):
    """Test that ``GBStructureTimeseries`` properly inherits from ``GBStructure``."""

    def test_is_subclass_of_gbstructure(self) -> None:
        """``GBStructureTimeseries`` must be a subclass of ``GBStructure``."""
        assert issubclass(GBStructureTimeseries, GBStructure)

    def test_instance_is_gbstructure(self) -> None:
        """A ``GBStructureTimeseries`` instance must also be a ``GBStructure``."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        assert isinstance(ts, GBStructure)


@pytest.mark.unit
class TestGBStructureTimeseriesInit(TestCase):
    """Test ``GBStructureTimeseries.__init__`` without real backends."""

    def _make_ts(
        self,
        backend: str = "ase",
        frames: int = 3,
        timestamps: list[int | float] | None = None,
    ) -> GBStructureTimeseries:
        """Return a ``GBStructureTimeseries`` with mocked frame data."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = backend
        ts.filename = "dummy.dump"
        mock_atoms = [MagicMock() for _ in range(frames)]
        ts.data = types.SimpleNamespace(atoms=mock_atoms, selection=[])
        ts.timestamps = timestamps
        return ts

    def test_timestamps_stored_when_provided(self) -> None:
        """Timestamps passed at init must be accessible as an attribute."""
        ts = self._make_ts(timestamps=[0, 10, 20])
        assert ts.timestamps == [0, 10, 20]

    def test_timestamps_none_by_default(self) -> None:
        """Timestamps must be ``None`` when not provided."""
        ts = self._make_ts()
        assert ts.timestamps is None

    def test_backend_attribute_preserved(self) -> None:
        """Backend must be stored as an attribute."""
        ts = self._make_ts(backend="ase")
        assert ts.backend == "ase"

    def test_init_raises_when_timestamps_len_mismatch_num_frames(self) -> None:
        """``__init__`` must raise when timestamps length and frame count differ."""
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "agility.analysis.GBStructure.__init__",
                MagicMock(return_value=None),
            )
            monkeypatch.setattr(
                GBStructureTimeseries,
                "num_frames",
                property(lambda _self: 3),
            )
            with pytest.raises(ValueError, match=r"len\(timestamps\)=2.*num_frames=3"):
                GBStructureTimeseries("ase", "dummy.dump", timestamps=[0, 10])

    def test_init_accepts_matching_timestamps_len(self) -> None:
        """``__init__`` must allow timestamps length matching frame count."""
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "agility.analysis.GBStructure.__init__",
                MagicMock(return_value=None),
            )
            monkeypatch.setattr(
                GBStructureTimeseries,
                "num_frames",
                property(lambda _self: 3),
            )
            ts = GBStructureTimeseries("ase", "dummy.dump", timestamps=[0, 10, 20])
        assert ts.timestamps == [0, 10, 20]


@pytest.mark.unit
class TestGBStructureTimeseriesNumFrames(TestCase):
    """Test the ``num_frames`` property without real backends."""

    def _make_ts_ase(self, n: int) -> GBStructureTimeseries:
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.data = types.SimpleNamespace(atoms=[MagicMock() for _ in range(n)], selection=[])
        return ts

    def test_num_frames_ase(self) -> None:
        """``num_frames`` must equal the number of ASE Atoms objects stored."""
        ts = self._make_ts_ase(5)
        assert ts.num_frames == 5

    def test_num_frames_unsupported_backend_raises(self) -> None:
        """``num_frames`` must raise ``NotImplementedError`` for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "pymatgen"
        with pytest.raises(NotImplementedError, match="pymatgen"):
            _ = ts.num_frames


@pytest.mark.unit
class TestGBStructureTimeseriesGetFrame(TestCase):
    """Test ``get_frame`` without real backends."""

    def _make_ts_ase(self, n: int) -> GBStructureTimeseries:
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.filename = "dummy.dump"
        mock_atoms = [MagicMock(name=f"frame_{i}") for i in range(n)]
        ts.data = types.SimpleNamespace(atoms=mock_atoms, selection=[])
        return ts

    def test_get_frame_returns_gbstructure(self) -> None:
        """``get_frame`` must return a ``GBStructure`` instance."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(0)
        assert isinstance(frame, GBStructure)

    def test_get_frame_correct_atoms(self) -> None:
        """``get_frame`` must set ``data.atoms`` to the correct frame's Atoms object."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(2)
        assert frame.data.atoms is ts.data.atoms[2]

    def test_get_frame_inherits_backend(self) -> None:
        """The returned ``GBStructure`` must have the same backend as the timeseries."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(1)
        assert frame.backend == "ase"

    def test_get_frame_unsupported_backend_raises(self) -> None:
        """``get_frame`` must raise ``NotImplementedError`` for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "pymatgen"
        with pytest.raises(NotImplementedError):
            ts.get_frame(0)

    def test_get_frame_ovito_clones_pipeline(self) -> None:
        """Ovito ``get_frame`` must clone the pipeline to avoid shared mutable state."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ovito"
        ts.filename = "dummy.dump"
        ts.pipeline = MagicMock()
        cloned_pipeline = MagicMock()
        cloned_pipeline.compute.return_value = MagicMock()
        ts.pipeline.clone.return_value = cloned_pipeline

        frame = ts.get_frame(1)

        ts.pipeline.clone.assert_called_once_with()
        cloned_pipeline.compute.assert_called_once_with(frame=1)
        assert frame.pipeline is cloned_pipeline

    def test_get_frame_negative_index_raises(self) -> None:
        """``get_frame`` must raise ``ValueError`` for a negative frame index."""
        ts = self._make_ts_ase(3)
        with pytest.raises(ValueError, match="non-negative"):
            ts.get_frame(-1)

    def test_get_frame_out_of_range_raises(self) -> None:
        """``get_frame`` must raise ``ValueError`` when the frame index is out of range."""
        ts = self._make_ts_ase(3)
        with pytest.raises(ValueError, match="out of range"):
            ts.get_frame(5)


@pytest.mark.unit
class TestGBStructureTimeseriesRemoveTimesteps(TestCase):
    """Test ``remove_timesteps`` without real backends."""

    def _make_ts_ase(
        self,
        n: int,
        timestamps: list[int | float] | None = None,
    ) -> GBStructureTimeseries:
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.filename = "dummy.dump"
        ts.data = types.SimpleNamespace(atoms=list(range(n)), selection=[])
        ts.timestamps = timestamps
        return ts

    def test_remove_timesteps_trims_frames(self) -> None:
        """``remove_timesteps`` must discard the first N frames."""
        ts = self._make_ts_ase(5)
        ts.remove_timesteps(2)
        assert ts.data.atoms == [2, 3, 4]

    def test_remove_timesteps_trims_timestamps(self) -> None:
        """``remove_timesteps`` must trim the timestamps list when it is set."""
        ts = self._make_ts_ase(5, timestamps=[0, 10, 20, 30, 40])
        ts.remove_timesteps(2)
        assert ts.timestamps == [20, 30, 40]

    def test_remove_timesteps_no_timestamps(self) -> None:
        """``remove_timesteps`` must not fail when timestamps is ``None``."""
        ts = self._make_ts_ase(4)
        ts.remove_timesteps(1)
        assert ts.data.atoms == [1, 2, 3]
        assert ts.timestamps is None

    def test_remove_timesteps_unsupported_backend_raises(self) -> None:
        """``remove_timesteps`` must raise ``NotImplementedError`` for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "lammps"
        with pytest.raises(NotImplementedError):
            ts.remove_timesteps(1)

    def test_remove_timesteps_negative_raises(self) -> None:
        """``remove_timesteps`` must raise ``ValueError`` for a negative count."""
        ts = self._make_ts_ase(5)
        with pytest.raises(ValueError, match="non-negative"):
            ts.remove_timesteps(-1)

    def test_remove_timesteps_exceeds_frames_raises(self) -> None:
        """``remove_timesteps`` must raise ``ValueError`` when count exceeds frame count."""
        ts = self._make_ts_ase(3)
        with pytest.raises(ValueError, match="exceeds"):
            ts.remove_timesteps(5)


# ---------------------------------------------------------------------------
# GBStructureTimeseries.read_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGBStructureTimeseriesReadFile(TestCase):
    """Test ``GBStructureTimeseries.read_file`` for the ase backend."""

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_read_file_ase_returns_list_of_frames(self) -> None:
        """The ase backend must read all frames into a list of Atoms objects."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.data = None
        mock_atoms_1 = MagicMock()
        mock_atoms_2 = MagicMock()
        with patch("ase.io.read") as mock_read:
            mock_read.return_value = [mock_atoms_1, mock_atoms_2]
            ts.read_file("traj.xyz")
            mock_read.assert_called_once()
            assert ts.data.atoms == [mock_atoms_1, mock_atoms_2]
            assert ts.data.selection == []

    @pytest.mark.skipif(not find_spec("ase"), reason="ase not installed")
    def test_read_file_ase_wraps_single_frame_in_list(self) -> None:
        """When ``ase.io.read`` returns a single Atoms object, it must be wrapped in a list."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.data = None
        mock_atoms = MagicMock()
        with patch("ase.io.read") as mock_read:
            mock_read.return_value = mock_atoms
            ts.read_file("traj.xyz")
            assert ts.data.atoms == [mock_atoms]

    def test_read_file_non_ase_delegates_to_super(self) -> None:
        """Non-ase backends must delegate to ``GBStructure.read_file``."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ovito"
        ts.data = None
        with patch.object(GBStructure, "read_file") as mock_super_read:
            ts.read_file("traj.dump")
            mock_super_read.assert_called_once_with("traj.dump")

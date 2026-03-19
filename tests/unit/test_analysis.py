"""Unit tests for analysis.py using mock objects — no real backends required."""

from __future__ import annotations

import types
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from agility.analysis import GBStructure, GBStructureTimeseries, not_implemented


@pytest.mark.unit
class TestNotImplemented(TestCase):
    """Test the not_implemented helper function."""

    def test_returns_not_implemented_error(self) -> None:
        """Test that not_implemented returns a NotImplementedError instance."""
        err = not_implemented("babel")
        assert isinstance(err, NotImplementedError)

    def test_message_contains_backend_name(self) -> None:
        """Test that the error message includes the backend name."""
        backend = "pyiron"
        err = not_implemented(backend)
        assert backend in str(err)

    def test_can_be_raised(self) -> None:
        """Test that the returned error can be raised."""
        backend = "babel"
        with pytest.raises(NotImplementedError, match=backend):
            raise not_implemented(backend)


@pytest.mark.unit
class TestSaveStructureLammps(TestCase):
    """Test save_structure for the lammps backend using a mock LAMMPS object."""

    def setUp(self) -> None:
        """Set up a GBStructure with a mocked pylmp."""
        self.gbs = GBStructure.__new__(GBStructure)
        self.gbs.backend = "lammps"
        self.gbs.pylmp = MagicMock()

    def test_save_structure_invalid_file_type_raises_value_error(self) -> None:
        """Test that save_structure raises ValueError for an unknown file type."""
        with pytest.raises(ValueError, match="Unrecognised file type"):
            self.gbs.save_structure("out.xyz", "xyz")

    def test_save_structure_data_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_data for file_type='data'."""
        self.gbs.save_structure("out.lmp", "data")
        self.gbs.pylmp.write_data.assert_called_once_with("out.lmp")

    def test_save_structure_dump_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_dump for file_type='dump'."""
        self.gbs.save_structure("out.dump", "dump")
        self.gbs.pylmp.write_dump.assert_called_once_with("out.dump")

    def test_save_structure_restart_delegates_to_pylmp(self) -> None:
        """Test that save_structure calls write_restart for file_type='restart'."""
        self.gbs.save_structure("out.restart", "restart")
        self.gbs.pylmp.write_restart.assert_called_once_with("out.restart")


@pytest.mark.unit
class TestClearSelectionUnit(TestCase):
    """Test _clear_selection using mock data objects — no backend packages required."""

    def _make_gbs(self, backend: str, selection: list[int]) -> GBStructure:
        gbs = GBStructure.__new__(GBStructure)
        gbs.backend = backend
        gbs.data = types.SimpleNamespace(selection=list(selection))
        return gbs

    def test_clear_selection_pymatgen(self) -> None:
        """Test that _clear_selection resets the selection list for the pymatgen backend."""
        gbs = self._make_gbs("pymatgen", [0, 1, 2])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_selection_ase(self) -> None:
        """Test that _clear_selection resets the selection list for the ase backend."""
        gbs = self._make_gbs("ase", [3, 5, 7])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_clear_already_empty_selection(self) -> None:
        """Test that _clear_selection is a no-op when the selection is already empty."""
        gbs = self._make_gbs("pymatgen", [])
        gbs._clear_selection()  # noqa: SLF001
        assert gbs.data.selection == []


@pytest.mark.unit
class TestInvertSelectionUnit(TestCase):
    """Test _invert_selection using mock data objects — no backend packages required."""

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
        """Test that inverting an empty selection selects all sites (pymatgen backend)."""
        gbs = self._make_gbs("pymatgen", [], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [0, 1, 2, 3]

    def test_invert_full_selection_ase(self) -> None:
        """Test that inverting a full selection yields an empty selection (ase backend)."""
        gbs = self._make_gbs("ase", [0, 1, 2, 3], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == []

    def test_invert_partial_selection(self) -> None:
        """Test that inverting a partial selection returns the complementary set."""
        gbs = self._make_gbs("pymatgen", [0, 2], 4)
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == [1, 3]

    def test_invert_twice_restores_original(self) -> None:
        """Test that inverting a selection twice restores the original selection."""
        original = [1, 3]
        gbs = self._make_gbs("ase", original, 4)
        gbs._invert_selection()  # noqa: SLF001
        gbs._invert_selection()  # noqa: SLF001
        assert gbs.data.selection == original


@pytest.mark.unit
class TestGBStructureTimeseriesInheritance(TestCase):
    """Test that GBStructureTimeseries properly inherits from GBStructure."""

    def test_is_subclass_of_gbstructure(self) -> None:
        """GBStructureTimeseries must be a subclass of GBStructure."""
        assert issubclass(GBStructureTimeseries, GBStructure)

    def test_instance_is_gbstructure(self) -> None:
        """A GBStructureTimeseries instance must also be a GBStructure."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        assert isinstance(ts, GBStructure)


@pytest.mark.unit
class TestGBStructureTimeseriesInit(TestCase):
    """Test GBStructureTimeseries.__init__ without real backends."""

    def _make_ts(
        self,
        backend: str = "ase",
        frames: int = 3,
        timestamps: list[int | float] | None = None,
    ) -> GBStructureTimeseries:
        """Return a GBStructureTimeseries with mocked frame data."""
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
        """Timestamps must be None when not provided."""
        ts = self._make_ts()
        assert ts.timestamps is None

    def test_backend_attribute_preserved(self) -> None:
        """Backend must be stored as an attribute."""
        ts = self._make_ts(backend="ase")
        assert ts.backend == "ase"


@pytest.mark.unit
class TestGBStructureTimeseriesNumFrames(TestCase):
    """Test the num_frames property without real backends."""

    def _make_ts_ase(self, n: int) -> GBStructureTimeseries:
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.data = types.SimpleNamespace(atoms=[MagicMock() for _ in range(n)], selection=[])
        return ts

    def test_num_frames_ase(self) -> None:
        """num_frames must equal the number of ASE Atoms objects stored."""
        ts = self._make_ts_ase(5)
        assert ts.num_frames == 5

    def test_num_frames_unsupported_backend_raises(self) -> None:
        """num_frames must raise NotImplementedError for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "pymatgen"
        with pytest.raises(NotImplementedError, match="pymatgen"):
            _ = ts.num_frames


@pytest.mark.unit
class TestGBStructureTimeseriesGetFrame(TestCase):
    """Test get_frame without real backends."""

    def _make_ts_ase(self, n: int) -> GBStructureTimeseries:
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "ase"
        ts.filename = "dummy.dump"
        mock_atoms = [MagicMock(name=f"frame_{i}") for i in range(n)]
        ts.data = types.SimpleNamespace(atoms=mock_atoms, selection=[])
        return ts

    def test_get_frame_returns_gbstructure(self) -> None:
        """get_frame must return a GBStructure instance."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(0)
        assert isinstance(frame, GBStructure)

    def test_get_frame_correct_atoms(self) -> None:
        """get_frame must set data.atoms to the correct frame's Atoms object."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(2)
        assert frame.data.atoms is ts.data.atoms[2]

    def test_get_frame_inherits_backend(self) -> None:
        """The returned GBStructure must have the same backend as the timeseries."""
        ts = self._make_ts_ase(3)
        frame = ts.get_frame(1)
        assert frame.backend == "ase"

    def test_get_frame_unsupported_backend_raises(self) -> None:
        """get_frame must raise NotImplementedError for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "pymatgen"
        with pytest.raises(NotImplementedError):
            ts.get_frame(0)


@pytest.mark.unit
class TestGBStructureTimeseriesRemoveTimesteps(TestCase):
    """Test remove_timesteps without real backends."""

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
        """remove_timesteps must discard the first N frames."""
        ts = self._make_ts_ase(5)
        ts.remove_timesteps(2)
        assert ts.data.atoms == [2, 3, 4]

    def test_remove_timesteps_trims_timestamps(self) -> None:
        """remove_timesteps must trim the timestamps list when it is set."""
        ts = self._make_ts_ase(5, timestamps=[0, 10, 20, 30, 40])
        ts.remove_timesteps(2)
        assert ts.timestamps == [20, 30, 40]

    def test_remove_timesteps_no_timestamps(self) -> None:
        """remove_timesteps must not fail when timestamps is None."""
        ts = self._make_ts_ase(4)
        ts.remove_timesteps(1)
        assert ts.data.atoms == [1, 2, 3]
        assert ts.timestamps is None

    def test_remove_timesteps_unsupported_backend_raises(self) -> None:
        """remove_timesteps must raise NotImplementedError for unsupported backends."""
        ts = GBStructureTimeseries.__new__(GBStructureTimeseries)
        ts.backend = "lammps"
        with pytest.raises(NotImplementedError):
            ts.remove_timesteps(1)

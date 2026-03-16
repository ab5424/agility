"""Unit tests for polycrystal.py using mock objects — no atomsk required."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

from agility.polycrystal import GrainDefinition, PolycrystalBuilder, find_atomsk


@pytest.mark.unit
class TestFindAtomsk(TestCase):
    """Test the find_atomsk() helper function."""

    @patch("agility.polycrystal.shutil.which", return_value="/usr/bin/atomsk")
    def test_returns_path_when_in_system_path(self, mock_which: MagicMock) -> None:
        """Test that find_atomsk returns the binary path when atomsk is on PATH."""
        result = find_atomsk()
        assert result == "/usr/bin/atomsk"
        mock_which.assert_called_once_with("atomsk")

    @patch("agility.polycrystal.shutil.which", return_value=None)
    @patch.object(pathlib.Path, "is_file", return_value=False)
    def test_returns_none_when_not_found(
        self,
        mock_is_file: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """Test that find_atomsk returns None when atomsk is not installed anywhere."""
        result = find_atomsk()
        assert result is None
        mock_which.assert_called_once_with("atomsk")
        mock_is_file.assert_called()


@pytest.mark.unit
class TestPolycrystalBuilderInit(TestCase):
    """Test PolycrystalBuilder initialisation."""

    def test_init_with_explicit_atomsk_path(self) -> None:
        """Test construction succeeds when an explicit atomsk path is supplied."""
        builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")
        assert builder._atomsk == "/usr/bin/atomsk"  # noqa: SLF001
        assert builder.unit_cell == pathlib.Path("unit.lmp")

    @patch("agility.polycrystal.find_atomsk", return_value=None)
    def test_raises_when_atomsk_not_found(self, mock_find: MagicMock) -> None:
        """Test that FileNotFoundError is raised when atomsk cannot be found."""
        with pytest.raises(FileNotFoundError, match="atomsk"):
            PolycrystalBuilder("unit.lmp")
        mock_find.assert_called_once()

    @patch("agility.polycrystal.find_atomsk", return_value="/usr/bin/atomsk")
    def test_uses_auto_detected_atomsk_path(self, mock_find: MagicMock) -> None:
        """Test that the auto-detected atomsk path is stored on the builder."""
        builder = PolycrystalBuilder("unit.lmp")
        assert builder._atomsk == "/usr/bin/atomsk"  # noqa: SLF001
        mock_find.assert_called_once()


@pytest.mark.unit
class TestPolycrystalBuilderConfiguration(TestCase):
    """Test PolycrystalBuilder grain and box configuration methods."""

    def setUp(self) -> None:
        """Set up a builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")

    def test_set_box(self) -> None:
        """Test that set_box stores the correct dimensions."""
        self.builder.set_box(100.0, 200.0, 150.0)
        assert self.builder._box == (100.0, 200.0, 150.0)  # noqa: SLF001

    def test_add_grain_appends_definition(self) -> None:
        """Test that add_grain appends a GrainDefinition to the internal list."""
        self.builder.add_grain((10.0, 20.0, 30.0), (0.0, 45.0, 90.0))
        assert len(self.builder._grains) == 1  # noqa: SLF001
        grain = self.builder._grains[0]  # noqa: SLF001
        assert grain.seed == (10.0, 20.0, 30.0)
        assert grain.euler_angles == (0.0, 45.0, 90.0)

    def test_add_multiple_grains(self) -> None:
        """Test that multiple explicit grains can be added."""
        self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.builder.add_grain((50.0, 50.0, 50.0), (30.0, 0.0, 0.0))
        assert len(self.builder._grains) == 2  # noqa: SLF001

    def test_set_random_grains_stores_count(self) -> None:
        """Test that set_random_grains stores the requested grain count."""
        self.builder.set_random_grains(5)
        assert self.builder._random_grains == 5  # noqa: SLF001

    def test_add_grain_raises_after_set_random(self) -> None:
        """Test that add_grain raises ValueError after set_random_grains is called."""
        self.builder.set_random_grains(3)
        with pytest.raises(ValueError, match="set_random_grains"):
            self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_set_random_raises_after_add_grain(self) -> None:
        """Test that set_random_grains raises ValueError after add_grain is called."""
        self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="add_grain"):
            self.builder.set_random_grains(3)

    def test_grains_property_returns_copy(self) -> None:
        """Test that the grains property returns a shallow copy of the list."""
        self.builder.add_grain((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        grains_copy = self.builder.grains
        assert len(grains_copy) == 1
        # Mutating the copy must not affect the internal list
        grains_copy.clear()
        assert len(self.builder._grains) == 1  # noqa: SLF001


@pytest.mark.unit
class TestPolycrystalBuilderWriteParamFile(TestCase):
    """Test the content produced by _write_param_file."""

    def setUp(self) -> None:
        """Set up a builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")

    def test_raises_without_box(self) -> None:
        """Test that _write_param_file raises ValueError when no box is set."""
        self.builder.set_random_grains(2)
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp,
            pytest.raises(ValueError, match="set_box"),
        ):
            self.builder._write_param_file(pathlib.Path(tmp.name))  # noqa: SLF001

    def test_raises_without_grains(self) -> None:
        """Test that _write_param_file raises ValueError when no grains are defined."""
        self.builder.set_box(100.0, 100.0, 100.0)
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp,
            pytest.raises(ValueError, match="add_grain"),
        ):
            self.builder._write_param_file(pathlib.Path(tmp.name))  # noqa: SLF001

    def test_random_grain_file_content(self) -> None:
        """Test that the parameter file for random grains has the correct content."""
        self.builder.set_box(100.0, 200.0, 300.0)
        self.builder.set_random_grains(4)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.builder._write_param_file(tmp_path)  # noqa: SLF001
            content = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)
        assert "box 100.0 200.0 300.0" in content
        assert "random 4" in content

    def test_explicit_grain_file_content(self) -> None:
        """Test that the parameter file for explicit grains has the correct content."""
        self.builder.set_box(50.0, 50.0, 50.0)
        self.builder.add_grain((10.0, 20.0, 30.0), (0.0, 45.0, 90.0))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.builder._write_param_file(tmp_path)  # noqa: SLF001
            content = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)
        assert "box 50.0 50.0 50.0" in content
        assert "grain 10.0 20.0 30.0" in content
        assert "0.0 45.0 90.0" in content

    def test_multiple_explicit_grains_all_written(self) -> None:
        """Test that all added explicit grains appear in the parameter file."""
        self.builder.set_box(100.0, 100.0, 100.0)
        self.builder.add_grain((25.0, 50.0, 50.0), (0.0, 0.0, 0.0))
        self.builder.add_grain((75.0, 50.0, 50.0), (45.0, 0.0, 0.0))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.builder._write_param_file(tmp_path)  # noqa: SLF001
            content = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)
        assert content.count("grain") == 2


@pytest.mark.unit
class TestPolycrystalBuilderBuild(TestCase):
    """Test the build() method subprocess invocation."""

    def setUp(self) -> None:
        """Set up a fully configured builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")
        self.builder.set_box(100.0, 100.0, 100.0)
        self.builder.set_random_grains(2)

    @patch("subprocess.run")
    def test_build_calls_subprocess(self, mock_run: MagicMock) -> None:
        """Test that build() invokes subprocess.run with the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("output.lmp")
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/atomsk"
        assert "--polycrystal" in cmd
        assert isinstance(result, pathlib.Path)

    @patch("subprocess.run")
    def test_build_includes_unit_cell_in_command(self, mock_run: MagicMock) -> None:
        """Test that the unit cell path appears in the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp")
        cmd = mock_run.call_args[0][0]
        assert "unit.lmp" in cmd

    @patch("subprocess.run")
    def test_build_passes_output_format(self, mock_run: MagicMock) -> None:
        """Test that the output_format argument is appended to the command."""
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp", output_format="lmp")
        cmd = mock_run.call_args[0][0]
        assert "lmp" in cmd

    @patch("subprocess.run")
    def test_build_passes_extra_options(self, mock_run: MagicMock) -> None:
        """Test that extra_options flags are forwarded to the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp", extra_options=["-overwrite"])
        cmd = mock_run.call_args[0][0]
        assert "-overwrite" in cmd

    @patch("subprocess.run")
    def test_build_returns_path_object(self, mock_run: MagicMock) -> None:
        """Test that build() returns a pathlib.Path instance."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("output.lmp")
        assert isinstance(result, pathlib.Path)

    @patch("subprocess.run")
    def test_build_without_format_no_format_arg(self, mock_run: MagicMock) -> None:
        """Test that no format keyword is appended when output_format is None.

        Expected command: [atomsk, --polycrystal, <unit_cell>, <param_file>, <prefix>]
        — exactly five elements with no trailing format identifier.
        """
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp")
        cmd = mock_run.call_args[0][0]
        assert len(cmd) == 5

    @patch("subprocess.run")
    def test_build_returns_format_extension_when_output_format_given(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that build() returns the path with the output_format extension.

        atomsk writes <prefix>.<output_format>, so the returned Path must reflect that
        even when output_file carries a different (or no) extension.
        """
        mock_run.return_value = MagicMock(returncode=0)
        # Pass a file with a .cfg extension but request lmp format
        result = self.builder.build("output.cfg", output_format="lmp")
        assert result.suffix == ".lmp"

    @patch("subprocess.run")
    def test_build_returns_correct_path_without_extension(self, mock_run: MagicMock) -> None:
        """Test that build() appends output_format when output_file has no extension."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("poly", output_format="lmp")
        assert result.suffix == ".lmp"
        assert result.stem == "poly"


@pytest.mark.unit
class TestGrainDefinition(TestCase):
    """Test the GrainDefinition dataclass."""

    def test_creation(self) -> None:
        """Test that a GrainDefinition stores seed and Euler angles correctly."""
        grain = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(10.0, 20.0, 30.0))
        assert grain.seed == (1.0, 2.0, 3.0)
        assert grain.euler_angles == (10.0, 20.0, 30.0)

    def test_equality(self) -> None:
        """Test that two GrainDefinitions with equal fields compare as equal."""
        g1 = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(0.0, 0.0, 0.0))
        g2 = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(0.0, 0.0, 0.0))
        assert g1 == g2

    def test_inequality(self) -> None:
        """Test that two GrainDefinitions with different fields are not equal."""
        g1 = GrainDefinition(seed=(0.0, 0.0, 0.0), euler_angles=(0.0, 0.0, 0.0))
        g2 = GrainDefinition(seed=(1.0, 0.0, 0.0), euler_angles=(0.0, 0.0, 0.0))
        assert g1 != g2

"""Unit tests for polycrystal.py using mock objects — no atomsk required."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

from agility.polycrystal import (
    GrainDefinition,
    PolycrystalBuilder,
    build_atomsk_from_source,
    find_atomsk,
)

# ---------------------------------------------------------------------------
# find_atomsk
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindAtomsk(TestCase):
    """Test the ``find_atomsk()`` helper function."""

    @patch("agility.polycrystal.shutil.which", return_value="/usr/bin/atomsk")
    def test_returns_path_when_in_system_path(self, mock_which: MagicMock) -> None:
        """``find_atomsk`` must return the binary path when atomsk is on PATH."""
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
        """``find_atomsk`` must return ``None`` when atomsk is not installed anywhere."""
        result = find_atomsk()
        assert result is None
        mock_which.assert_called_once_with("atomsk")
        mock_is_file.assert_called()

    @patch("agility.polycrystal.shutil.which", return_value=None)
    @patch.object(pathlib.Path, "is_file", return_value=True)
    @patch("agility.polycrystal.os.access", return_value=True)
    def test_returns_local_bin_when_not_on_path(
        self,
        mock_access: MagicMock,  # noqa: ARG002
        mock_is_file: MagicMock,  # noqa: ARG002
        mock_which: MagicMock,  # noqa: ARG002
    ) -> None:
        """``find_atomsk`` must fall back to ``~/.local/bin/atomsk`` when not on PATH."""
        result = find_atomsk()
        assert result is not None
        assert result.endswith("atomsk")


# ---------------------------------------------------------------------------
# build_atomsk_from_source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildAtomskFromSource(TestCase):
    """Test the ``build_atomsk_from_source`` function using mocked subprocess calls."""

    @patch("agility.polycrystal.shutil.which")
    def test_raises_when_gfortran_missing(self, mock_which: MagicMock) -> None:
        """``build_atomsk_from_source`` must raise ``RuntimeError`` when gfortran is missing."""
        mock_which.return_value = None
        with pytest.raises(RuntimeError, match="gfortran"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    def test_raises_when_git_missing(self, mock_which: MagicMock) -> None:
        """``build_atomsk_from_source`` must raise ``RuntimeError`` when git is missing."""
        mock_which.side_effect = lambda cmd: None if cmd == "git" else "/usr/bin/gfortran"
        with pytest.raises(RuntimeError, match="git"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    def test_raises_when_make_missing(self, mock_which: MagicMock) -> None:
        """``build_atomsk_from_source`` must raise ``RuntimeError`` when make is missing."""
        mock_which.side_effect = lambda cmd: None if cmd == "make" else "/usr/bin/gfortran"
        with pytest.raises(RuntimeError, match="make"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    @patch("agility.polycrystal.subprocess.run")
    @patch("agility.polycrystal.shutil.copy2")
    @patch.object(pathlib.Path, "is_file", return_value=True)
    def test_successful_build_returns_path(
        self,
        mock_is_file: MagicMock,  # noqa: ARG002
        mock_copy2: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """A successful build must return the path to the installed binary."""
        mock_which.return_value = "/usr/bin/gfortran"
        mock_run.return_value = MagicMock(returncode=0)

        def _copy2_side_effect(src: str, dst: str) -> None:  # noqa: ARG001
            pathlib.Path(dst).write_text("dummy", encoding="utf-8")

        mock_copy2.side_effect = _copy2_side_effect
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_atomsk_from_source(install_dir=tmpdir)
            assert result.endswith("atomsk")
            assert pathlib.Path(result).parent == pathlib.Path(tmpdir)

    @patch("agility.polycrystal.shutil.which")
    @patch("agility.polycrystal.subprocess.run")
    def test_clone_failure_raises_runtime_error(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """A failed ``git clone`` must raise ``RuntimeError``."""
        import subprocess as sp  # noqa: PLC0415

        mock_which.return_value = "/usr/bin/gfortran"
        mock_run.side_effect = sp.CalledProcessError(1, "git", stderr="clone failed")
        with pytest.raises(RuntimeError, match="Failed to clone"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    @patch("agility.polycrystal.subprocess.run")
    def test_make_failure_raises_runtime_error(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """A failed ``make`` must raise ``RuntimeError``."""
        import subprocess as sp  # noqa: PLC0415

        mock_which.return_value = "/usr/bin/gfortran"
        # First call (git clone) succeeds, second call (make) fails
        mock_run.side_effect = [
            MagicMock(returncode=0),
            sp.CalledProcessError(1, "make", stderr="make failed"),
        ]
        with pytest.raises(RuntimeError, match="Failed to compile"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    @patch("agility.polycrystal.subprocess.run")
    @patch.object(pathlib.Path, "is_file", return_value=False)
    def test_binary_not_found_after_build_raises(
        self,
        mock_is_file: MagicMock,  # noqa: ARG002
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """If the binary is missing after a successful build, ``FileNotFoundError`` must be raised."""  # noqa: E501
        mock_which.return_value = "/usr/bin/gfortran"
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(FileNotFoundError, match="binary not found"):
            build_atomsk_from_source()

    @patch("agility.polycrystal.shutil.which")
    @patch("agility.polycrystal.subprocess.run")
    @patch("agility.polycrystal.shutil.copy2")
    @patch.object(pathlib.Path, "is_file", return_value=True)
    def test_default_install_dir_uses_home_local_bin(
        self,
        mock_is_file: MagicMock,  # noqa: ARG002
        mock_copy2: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """When ``install_dir`` is ``None``, the default ``~/.local/bin`` must be used."""
        mock_which.return_value = "/usr/bin/gfortran"
        mock_run.return_value = MagicMock(returncode=0)

        def _copy2_side_effect(src: str, dst: str) -> None:  # noqa: ARG001
            pathlib.Path(dst).write_text("dummy", encoding="utf-8")

        mock_copy2.side_effect = _copy2_side_effect
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(pathlib.Path, "home", return_value=pathlib.Path(tmpdir)),
        ):
            result = build_atomsk_from_source()
            expected = pathlib.Path(tmpdir) / ".local" / "bin" / "atomsk"
            assert pathlib.Path(result) == expected


# ---------------------------------------------------------------------------
# GrainDefinition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGrainDefinition(TestCase):
    """Test the ``GrainDefinition`` dataclass."""

    def test_creation(self) -> None:
        """A ``GrainDefinition`` must store seed and Euler angles correctly."""
        grain = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(10.0, 20.0, 30.0))
        assert grain.seed == (1.0, 2.0, 3.0)
        assert grain.euler_angles == (10.0, 20.0, 30.0)

    def test_equality(self) -> None:
        """Two ``GrainDefinition`` instances with equal fields must compare as equal."""
        g1 = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(0.0, 0.0, 0.0))
        g2 = GrainDefinition(seed=(1.0, 2.0, 3.0), euler_angles=(0.0, 0.0, 0.0))
        assert g1 == g2

    def test_inequality(self) -> None:
        """Two ``GrainDefinition`` instances with different fields must not be equal."""
        g1 = GrainDefinition(seed=(0.0, 0.0, 0.0), euler_angles=(0.0, 0.0, 0.0))
        g2 = GrainDefinition(seed=(1.0, 0.0, 0.0), euler_angles=(0.0, 0.0, 0.0))
        assert g1 != g2


# ---------------------------------------------------------------------------
# PolycrystalBuilder.__init__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolycrystalBuilderInit(TestCase):
    """Test ``PolycrystalBuilder`` initialisation."""

    def test_init_with_explicit_atomsk_path(self) -> None:
        """Construction must succeed when an explicit atomsk path is supplied."""
        builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")
        assert builder._atomsk == "/usr/bin/atomsk"  # noqa: SLF001
        assert builder.unit_cell == pathlib.Path("unit.lmp").resolve()

    @patch("agility.polycrystal.find_atomsk", return_value=None)
    def test_raises_when_atomsk_not_found(self, mock_find: MagicMock) -> None:
        """``FileNotFoundError`` must be raised when atomsk cannot be found."""
        with pytest.raises(FileNotFoundError, match="atomsk"):
            PolycrystalBuilder("unit.lmp")
        mock_find.assert_called_once()

    @patch("agility.polycrystal.find_atomsk", return_value="/usr/bin/atomsk")
    def test_uses_auto_detected_atomsk_path(self, mock_find: MagicMock) -> None:
        """The auto-detected atomsk path must be stored on the builder."""
        builder = PolycrystalBuilder("unit.lmp")
        assert builder._atomsk == "/usr/bin/atomsk"  # noqa: SLF001
        mock_find.assert_called_once()

    def test_init_accepts_pathlib_unit_cell(self) -> None:
        """The unit cell must be accepted as a ``pathlib.Path``."""
        builder = PolycrystalBuilder(pathlib.Path("unit.lmp"), atomsk_path="/usr/bin/atomsk")
        assert builder.unit_cell == pathlib.Path("unit.lmp").resolve()

    def test_init_initialises_empty_state(self) -> None:
        """The builder must start with no box and no grains."""
        builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")
        assert builder._box is None  # noqa: SLF001
        assert builder._grains == []  # noqa: SLF001
        assert builder._random_grains is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# PolycrystalBuilder configuration methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolycrystalBuilderConfiguration(TestCase):
    """Test ``PolycrystalBuilder`` grain and box configuration methods."""

    def setUp(self) -> None:
        """Set up a builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")

    def test_set_box(self) -> None:
        """``set_box`` must store the correct dimensions."""
        self.builder.set_box(100.0, 200.0, 150.0)
        assert self.builder._box == (100.0, 200.0, 150.0)  # noqa: SLF001

    def test_add_grain_appends_definition(self) -> None:
        """``add_grain`` must append a ``GrainDefinition`` to the internal list."""
        self.builder.add_grain((10.0, 20.0, 30.0), (0.0, 45.0, 90.0))
        assert len(self.builder._grains) == 1  # noqa: SLF001
        grain = self.builder._grains[0]  # noqa: SLF001
        assert grain.seed == (10.0, 20.0, 30.0)
        assert grain.euler_angles == (0.0, 45.0, 90.0)

    def test_add_multiple_grains(self) -> None:
        """Multiple explicit grains can be added."""
        self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.builder.add_grain((50.0, 50.0, 50.0), (30.0, 0.0, 0.0))
        assert len(self.builder._grains) == 2  # noqa: SLF001

    def test_set_random_grains_stores_count(self) -> None:
        """``set_random_grains`` must store the requested grain count."""
        self.builder.set_random_grains(5)
        assert self.builder._random_grains == 5  # noqa: SLF001

    def test_add_grain_raises_after_set_random(self) -> None:
        """``add_grain`` must raise ``ValueError`` after ``set_random_grains`` is called."""
        self.builder.set_random_grains(3)
        with pytest.raises(ValueError, match="set_random_grains"):
            self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_set_random_raises_after_add_grain(self) -> None:
        """``set_random_grains`` must raise ``ValueError`` after ``add_grain`` is called."""
        self.builder.add_grain((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="add_grain"):
            self.builder.set_random_grains(3)

    def test_grains_property_returns_copy(self) -> None:
        """The ``grains`` property must return a shallow copy of the list."""
        self.builder.add_grain((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        grains_copy = self.builder.grains
        assert len(grains_copy) == 1
        # Mutating the copy must not affect the internal list
        grains_copy.clear()
        assert len(self.builder._grains) == 1  # noqa: SLF001


# ---------------------------------------------------------------------------
# PolycrystalBuilder._write_param_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolycrystalBuilderWriteParamFile(TestCase):
    """Test the content produced by ``_write_param_file``."""

    def setUp(self) -> None:
        """Set up a builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")

    def test_raises_without_box(self) -> None:
        """``_write_param_file`` must raise ``ValueError`` when no box is set."""
        self.builder.set_random_grains(2)
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp,
            pytest.raises(ValueError, match="set_box"),
        ):
            self.builder._write_param_file(pathlib.Path(tmp.name))  # noqa: SLF001

    def test_raises_without_grains(self) -> None:
        """``_write_param_file`` must raise ``ValueError`` when no grains are defined."""
        self.builder.set_box(100.0, 100.0, 100.0)
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp,
            pytest.raises(ValueError, match="add_grain"),
        ):
            self.builder._write_param_file(pathlib.Path(tmp.name))  # noqa: SLF001

    def test_random_grain_file_content(self) -> None:
        """The parameter file for random grains must have the correct content."""
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
        """The parameter file for explicit grains must have the correct content."""
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
        """All added explicit grains must appear in the parameter file."""
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


# ---------------------------------------------------------------------------
# PolycrystalBuilder.build
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolycrystalBuilderBuild(TestCase):
    """Test the ``build()`` method subprocess invocation."""

    def setUp(self) -> None:
        """Set up a fully configured builder with a mocked atomsk path."""
        self.builder = PolycrystalBuilder("unit.lmp", atomsk_path="/usr/bin/atomsk")
        self.builder.set_box(100.0, 100.0, 100.0)
        self.builder.set_random_grains(2)

    @patch("subprocess.run")
    def test_build_calls_subprocess(self, mock_run: MagicMock) -> None:
        """``build()`` must invoke ``subprocess.run`` with the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("output.lmp")
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/atomsk"
        assert "--polycrystal" in cmd
        assert isinstance(result, pathlib.Path)

    @patch("subprocess.run")
    def test_build_includes_unit_cell_in_command(self, mock_run: MagicMock) -> None:
        """The unit cell path must appear in the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            unit_cell = pathlib.Path(tmpdir) / "unit.lmp"
            unit_cell.write_text("dummy", encoding="utf-8")
            builder = PolycrystalBuilder(unit_cell, atomsk_path="/usr/bin/atomsk")
            builder.set_box(100.0, 100.0, 100.0)
            builder.set_random_grains(2)
            builder.build(pathlib.Path(tmpdir) / "output.lmp")

        cmd = mock_run.call_args[0][0]
        assert pathlib.Path(cmd[2]).name == "unit.lmp"
        assert pathlib.Path(cmd[2]).is_absolute()

    @patch("subprocess.run")
    def test_build_passes_output_format(self, mock_run: MagicMock) -> None:
        """The ``output_format`` argument must be appended to the command."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "output.lmp"
            output.write_text("dummy", encoding="utf-8")
            self.builder.build(output, output_format="lmp")
        cmd = mock_run.call_args[0][0]
        assert "lmp" in cmd

    @patch("subprocess.run")
    def test_build_passes_extra_options(self, mock_run: MagicMock) -> None:
        """``extra_options`` flags must be forwarded to the atomsk command."""
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp", extra_options=["-overwrite"])
        cmd = mock_run.call_args[0][0]
        assert "-overwrite" in cmd

    @patch("subprocess.run")
    def test_build_returns_path_object(self, mock_run: MagicMock) -> None:
        """``build()`` must return a ``pathlib.Path`` instance."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("output.lmp")
        assert isinstance(result, pathlib.Path)

    @patch("subprocess.run")
    def test_build_without_format_no_format_arg(self, mock_run: MagicMock) -> None:
        """No format keyword must be appended when ``output_format`` is ``None``."""
        mock_run.return_value = MagicMock(returncode=0)
        self.builder.build("output.lmp")
        cmd = mock_run.call_args[0][0]
        assert len(cmd) == 5
        assert pathlib.Path(cmd[-1]) == pathlib.Path("output.lmp").resolve()

    @patch("subprocess.run")
    def test_build_returns_requested_path_when_output_format_omitted(
        self,
        mock_run: MagicMock,
    ) -> None:
        """``build()`` must return the resolved requested path when ``output_format`` is omitted."""
        mock_run.return_value = MagicMock(returncode=0)
        result = self.builder.build("output.lmp")
        assert result == pathlib.Path("output.lmp").resolve()

    @patch("subprocess.run")
    def test_build_returns_format_extension_when_output_format_given(
        self,
        mock_run: MagicMock,
    ) -> None:
        """``build()`` must return the path with the ``output_format`` extension."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "output.cfg"
            formatted_output = pathlib.Path(tmpdir) / "output.lmp"
            formatted_output.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="lmp")
        assert result.suffix == ".lmp"

    @patch("subprocess.run")
    def test_build_returns_correct_path_without_extension(self, mock_run: MagicMock) -> None:
        """``build()`` must append ``output_format`` when ``output_file`` has no extension."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly"
            formatted_output = pathlib.Path(tmpdir) / "poly.lmp"
            formatted_output.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="lmp")
        assert result.suffix == ".lmp"
        assert result.stem == "poly"

    @patch("subprocess.run")
    def test_build_allows_compound_output_format(self, mock_run: MagicMock) -> None:
        """``build()`` must support compound output formats such as ``cfg.gz``."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.cfg"
            formatted_output = pathlib.Path(tmpdir) / "poly.cfg.gz"
            formatted_output.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="cfg.gz")
        assert str(result).endswith("poly.cfg.gz")

    @patch("subprocess.run")
    def test_build_strips_all_suffixes_when_format_given(self, mock_run: MagicMock) -> None:
        """All existing ``output_file`` suffixes must be stripped for the atomsk output prefix."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.cfg.gz"
            formatted_output = pathlib.Path(tmpdir) / "poly.lmp"
            formatted_output.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="lmp")

        cmd = mock_run.call_args[0][0]
        assert pathlib.Path(cmd[4]).name == "poly"
        assert result.name == "poly.lmp"

    @patch("subprocess.run")
    def test_build_returns_prefix_path_if_format_output_without_extension(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Fallback to output prefix when atomsk writes extensionless output (e.g. vasp)."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.vasp"
            prefix = pathlib.Path(tmpdir) / "poly"
            prefix.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="vasp")

        assert result == prefix.resolve()

    @patch("subprocess.run")
    def test_build_returns_poscar_if_vasp_output_is_poscar(self, mock_run: MagicMock) -> None:
        """Fallback to ``POSCAR`` when atomsk writes VASP output to ``POSCAR``."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.vasp"
            prefix = pathlib.Path(tmpdir) / "poly"
            poscar = pathlib.Path(tmpdir) / "POSCAR"
            poscar.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="vasp")

            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == "vasp"
            assert not output.exists()
            assert not prefix.exists()
            assert result == poscar.resolve()

    @patch("subprocess.run")
    def test_build_returns_contcar_if_vasp_output_is_contcar(self, mock_run: MagicMock) -> None:
        """Fallback to ``CONTCAR`` when atomsk writes VASP output to ``CONTCAR``."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.vasp"
            prefix = pathlib.Path(tmpdir) / "poly"
            contcar = pathlib.Path(tmpdir) / "CONTCAR"
            contcar.write_text("dummy", encoding="utf-8")
            result = self.builder.build(output, output_format="vasp")

            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == "vasp"
            assert not output.exists()
            assert not prefix.exists()
            assert not (pathlib.Path(tmpdir) / "POSCAR").exists()
            assert result == contcar.resolve()

    @patch("subprocess.run")
    def test_build_sets_cwd_to_output_directory(self, mock_run: MagicMock) -> None:
        """Atomsk must be executed in the output file directory."""
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.lmp"
            self.builder.build(output)

        assert mock_run.call_args.kwargs["cwd"] == str(output.resolve().parent)

    @patch("subprocess.run")
    def test_build_raises_if_formatted_output_file_not_found(self, mock_run: MagicMock) -> None:
        """``build()`` must raise when no expected output file exists."""
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "poly.vasp"
            with pytest.raises(FileNotFoundError, match="output file was not found"):
                self.builder.build(output, output_format="vasp")

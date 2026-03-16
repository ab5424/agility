# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Polycrystal structure generation using atomsk."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


def find_atomsk() -> str | None:
    """Find the atomsk executable on the system.

    Searches the system ``PATH`` first, then falls back to
    ``~/.local/bin/atomsk``.

    Returns:
        Absolute path to the atomsk binary, or ``None`` if not found.
    """
    path = shutil.which("atomsk")
    if path is not None:
        return path

    local_bin = pathlib.Path.home() / ".local" / "bin" / "atomsk"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    return None


def build_atomsk_from_source(
    install_dir: str | pathlib.Path | None = None,
) -> str:
    """Clone, compile, and install atomsk from source.

    Requires ``git``, ``make``, and a Fortran compiler (``gfortran``) to be
    available on the system.  An active internet connection is needed to
    clone the repository from GitHub.

    Args:
        install_dir: Directory where the compiled binary will be placed.
            Defaults to ``~/.local/bin``.

    Returns:
        Absolute path to the installed atomsk binary.

    Raises:
        RuntimeError: If a required build tool (``git`` or ``make``) is not
            found on ``PATH``, or if any build step fails.
        FileNotFoundError: If the compiled binary is not found after a
            successful build.
    """
    if install_dir is None:
        _install_dir = pathlib.Path.home() / ".local" / "bin"
    else:
        _install_dir = pathlib.Path(install_dir)

    _install_dir.mkdir(parents=True, exist_ok=True)

    gfortran_cmd = shutil.which("gfortran")
    if gfortran_cmd is None:
        msg = "gfortran is required to build atomsk from source but was not found on PATH."
        raise RuntimeError(msg)

    git_cmd = shutil.which("git")
    if git_cmd is None:
        msg = "git is required to build atomsk from source but was not found on PATH."
        raise RuntimeError(msg)

    make_cmd = shutil.which("make")
    if make_cmd is None:
        msg = "make is required to build atomsk from source but was not found on PATH."
        raise RuntimeError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dest = pathlib.Path(tmpdir) / "atomsk"

        try:
            subprocess.run(  # noqa: S603
                [
                    git_cmd,
                    "clone",
                    "--depth=1",
                    "https://github.com/pierrehirel/atomsk.git",
                    str(clone_dest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"Failed to clone atomsk repository: {exc.stderr}"
            raise RuntimeError(msg) from exc

        src_dir = clone_dest / "src"

        try:
            subprocess.run(  # noqa: S603
                [make_cmd],
                check=True,
                cwd=str(src_dir),
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"Failed to compile atomsk: {exc.stderr}"
            raise RuntimeError(msg) from exc

        binary_src = src_dir / "atomsk"
        if not binary_src.is_file():
            msg = f"Build completed but binary not found at {binary_src}."
            raise FileNotFoundError(msg)

        binary_dest = _install_dir / "atomsk"
        shutil.copy2(str(binary_src), str(binary_dest))
        binary_dest.chmod(0o755)

    return str(_install_dir / "atomsk")


@dataclass
class GrainDefinition:
    """Defines a single grain in a polycrystal.

    Attributes:
        seed: ``(x, y, z)`` coordinates of the grain nucleus in Angstroms.
        euler_angles: Bunge Euler angles ``(phi1, Phi, phi2)`` in degrees
            describing the grain orientation.
    """

    seed: tuple[float, float, float]
    euler_angles: tuple[float, float, float]


class PolycrystalBuilder:
    """Build polycrystalline structures using atomsk.

    Wraps the ``atomsk --polycrystal`` command so that grain seeds and
    crystal orientations can be configured from Python.  If atomsk is
    available on the system PATH it is used automatically.  Otherwise
    supply an explicit *atomsk_path* or call
    :func:`build_atomsk_from_source` first.

    Example::

        builder = PolycrystalBuilder("Al_fcc.lmp")
        builder.set_box(200.0, 200.0, 200.0)
        builder.add_grain((50.0, 100.0, 100.0), (0.0, 0.0, 0.0))
        builder.add_grain((150.0, 100.0, 100.0), (45.0, 0.0, 0.0))
        output_path = builder.build("polycrystal.lmp")

    Args:
        unit_cell: Path to the unit cell file recognised by atomsk (e.g.
            a LAMMPS data file, VASP POSCAR, or XYZ file).
        atomsk_path: Explicit path to the atomsk binary.  When ``None``
            :func:`find_atomsk` is called to locate it automatically.

    Raises:
        FileNotFoundError: If atomsk cannot be found and *atomsk_path* is
            not supplied.
    """

    def __init__(
        self,
        unit_cell: str | pathlib.Path,
        atomsk_path: str | pathlib.Path | None = None,
    ) -> None:
        """Initialize the builder with a unit cell and locate the atomsk binary.

        If *atomsk_path* is ``None``, :func:`find_atomsk` is called to search
        the system PATH and ``~/.local/bin``.
        """
        self.unit_cell = pathlib.Path(unit_cell)
        self._box: tuple[float, float, float] | None = None
        self._grains: list[GrainDefinition] = []
        self._random_grains: int | None = None

        if atomsk_path is not None:
            self._atomsk: str = str(atomsk_path)
        else:
            detected = find_atomsk()
            if detected is None:
                msg = (
                    "atomsk executable not found. Install atomsk "
                    "(https://atomsk.univ-lille.fr/) or call "
                    "agility.polycrystal.build_atomsk_from_source() to build it locally. "
                    "Note: Building from source requires git, make, and gfortran to be installed "
                    "and available on PATH."
                )
                raise FileNotFoundError(msg)
            self._atomsk = detected

    @property
    def grains(self) -> list[GrainDefinition]:
        """Return a copy of the currently defined grain list."""
        return list(self._grains)

    def set_box(self, lx: float, ly: float, lz: float) -> None:
        """Set the simulation box dimensions.

        Args:
            lx: Box length along *x* in Angstroms.
            ly: Box length along *y* in Angstroms.
            lz: Box length along *z* in Angstroms.
        """
        self._box = (lx, ly, lz)

    def add_grain(
        self,
        seed: tuple[float, float, float],
        euler_angles: tuple[float, float, float],
    ) -> None:
        """Add a grain with an explicit seed position and orientation.

        Cannot be combined with :meth:`set_random_grains`.

        Args:
            seed: ``(x, y, z)`` position of the grain nucleus in Angstroms.
            euler_angles: Bunge Euler angles ``(phi1, Phi, phi2)`` in degrees.

        Raises:
            ValueError: If :meth:`set_random_grains` has already been called.
        """
        if self._random_grains is not None:
            msg = "Cannot mix add_grain() with set_random_grains()."
            raise ValueError(msg)
        self._grains.append(GrainDefinition(seed=seed, euler_angles=euler_angles))

    def set_random_grains(self, n: int) -> None:
        """Request *n* randomly positioned and oriented grains.

        Cannot be combined with :meth:`add_grain`.

        Args:
            n: Number of grains to place randomly.

        Raises:
            ValueError: If grains have already been added via
                :meth:`add_grain`.
        """
        if self._grains:
            msg = "Cannot mix set_random_grains() with add_grain()."
            raise ValueError(msg)
        self._random_grains = n

    def _write_param_file(self, path: pathlib.Path) -> None:
        """Write the atomsk polycrystal parameter file to *path*.

        Args:
            path: Destination path for the parameter file.

        Raises:
            ValueError: If box dimensions or grain definitions are missing.
        """
        if self._box is None:
            msg = "Box dimensions must be set. Call set_box() before build()."
            raise ValueError(msg)

        if self._random_grains is None and not self._grains:
            msg = "No grains defined. Call add_grain() or set_random_grains() before build()."
            raise ValueError(msg)

        lines: list[str] = [f"box {self._box[0]} {self._box[1]} {self._box[2]}\n"]

        if self._random_grains is not None:
            lines.append(f"random {self._random_grains}\n")
        else:
            for grain in self._grains:
                x, y, z = grain.seed
                phi1, phi_cap, phi2 = grain.euler_angles
                lines.append(f"grain {x} {y} {z}  {phi1} {phi_cap} {phi2}\n")

        path.write_text("".join(lines), encoding="utf-8")

    def build(
        self,
        output_file: str | pathlib.Path,
        output_format: str | None = None,
        extra_options: list[str] | None = None,
    ) -> pathlib.Path:
        """Build the polycrystal structure by invoking atomsk.

        atomsk infers the output file format from the extension of
        *output_file*.  Pass *output_format* (e.g. ``"lmp"`` or
        ``"vasp"``) to override the format or when the extension alone is
        ambiguous.

        Args:
            output_file: Desired path for the generated structure file.  The
                file extension is used by atomsk to determine the output
                format unless *output_format* is given.
            output_format: atomsk format keyword appended to the command
                (e.g. ``"lmp"``, ``"vasp"``, ``"xyz"``).
            extra_options: Additional command-line flags forwarded verbatim
                to atomsk after the format keyword.

        Returns:
            Path to the generated structure file.

        Raises:
            ValueError: If box dimensions or grain definitions have not been
                set.
            subprocess.CalledProcessError: If atomsk exits with a non-zero
                return code.
        """
        output_path = pathlib.Path(output_file).resolve()

        # Determine both the argument passed to atomsk and the resulting output
        # path it will write.
        if output_format is not None:
            output_prefix_path = output_path
            while output_prefix_path.suffix:
                output_prefix_path = output_prefix_path.with_suffix("")

            output_prefix = str(output_prefix_path)
            output_arg = output_prefix
            actual_output = pathlib.Path(f"{output_prefix}.{output_format}")
        else:
            output_arg = str(output_path)
            actual_output = output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            param_file = pathlib.Path(tmpdir) / "polycrystal.txt"
            self._write_param_file(param_file)

            cmd: list[str] = [
                self._atomsk,
                "--polycrystal",
                str(self.unit_cell),
                str(param_file),
                output_arg,
            ]

            if output_format is not None:
                cmd.append(output_format)

            if extra_options is not None:
                cmd.extend(extra_options)

            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

        return actual_output

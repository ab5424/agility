# Copyright (c) Alexander Bonkowski.
# Distributed under the terms of the MIT License.

"""Minimiser for lammps, vasp, gulp."""
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    from lammps import lammps


def mimimise_lmp(
    lmp: lammps,
    style: str = "fire",
    min_opt: Sequence[Union[int, float]] = (0, 1e-8, 1000, 100000),
    mod: Optional[tuple] = None,
) -> lammps:
    """Run minimisation in lammps.

    Args:
        lmp: lammps instance for minimisation
        style: Minimisation style. Possible options: cg or hftn or sd or quickmin or fire or
        min_opt:
        fire/old or spin or spin/cg or spin/lbfgs.
        mod: list of modifications for

    Returns:
        lmp: lammps object with minimised structure.
    """
    if len(min_opt) != 4:
        print(
            "The list/tuple for minimisation must contain four arguments:"
            "etol, ftol, maxiter, maxeval."
        )
    lmp.min_style(f"{style}")
    if mod:
        for i in mod:
            lmp.min_modify(" ".join(i))
    lmp.minimize(f"{min_opt[0]} {min_opt[1]} {min_opt[2]} {min_opt[3]}")
    return lmp

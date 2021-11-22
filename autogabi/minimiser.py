# Copyright (c) Alexander Bonkowski.
# Distributed under the terms of the MIT License.

# Minimiser for lammps, vasp, gulp


def mimimise_lmp(
    lmp,
    style: str = "fire",
    etol: float = 0.0,
    ftol: float = 1e-8,
    maxiter: int = 1000,
    maxeval: int = 100000,
    mod: list = None,
):
    """Run mimimisation in lammps.

    Args:
        lmp: lammps instance for minimisation
        style: Mimimisation style. Possible options: cg or hftn or sd or quickmin or fire or
        fire/old or spin or spin/cg or spin/lbfgs.
        mod: list of modifications for

    Returns:
        lmp: lammps object with minimised structure.
    """
    lmp.min_style(f"{style}")
    if mod:
        for i in mod:
            lmp.min_modify(" ".join(i))
    lmp.minimize(f"{etol} {ftol} {maxiter} {maxeval}")

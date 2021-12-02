"""Minimiser for lammps, vasp, gulp."""

# Copyright (c) Alexander Bonkowski.
# Distributed under the terms of the MIT License.


def mimimise_lmp(
    lmp,
    style: str = "fire",
    min: tuple = (0.0, 1e-8, 1000, 100000),
    mod: tuple = None,
    mpi: bool = True,
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
    if len(min) != 4:
        print(
            "The list/tuple for minimisation must contain four arguments:"
            "etol, ftol, maxiter, maxeval."
        )
    if mpi:
        from mpi4py import MPI

        me = MPI.COMM_WORLD.Get_rank()
        nprocs = MPI.COMM_WORLD.Get_size()
    lmp.min_style(f"{style}")
    if mod:
        for i in mod:
            lmp.min_modify(" ".join(i))
    lmp.minimize(f"{min[0]} {min[1]} {min[2]} {min[3]}")
    return lmp

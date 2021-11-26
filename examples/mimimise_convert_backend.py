import numpy as np
from autogabi.analysis import GBStructure
from autogabi.plotting import render_ovito

gb = GBStructure('lammps', 'LSF_supercell_md3.lmp', pair_style='buck/coul/long 12', kspace_style='pppm 1.0e-4')

gb.pylmp.thermo(5)
gb.minimise()


# Get grain orientation

# https://github.com/paulhof/GraDe-A
# Centrosymetric parameter


# Polyhedral template matching is implemented in ovito, ASAP, and LAMMPS
# P. M. Larsen, S. Schmidt, J. Schiøtz, Modelling Simul. Mater. Sci. Eng. 2016, 24, 055007.
# https://docs.lammps.org/compute_ptm_atom.html

def get_grain_orientation(backend):
    if backend == 'ovito':
        # First, perform PTM
        # Iterate over all ions to get grains
        #   Pick one fcc ion, if > 2 ffc neighbors, count to grain, otherwise disregard
        # Get average

        return None

# from pymatgen.io.lammps.data import LammpsData
import math

from ovito.plugins.ParticlesPython import VoronoiAnalysisModifier
from ovito.plugins.PyScript import Viewport
from ovito.plugins.StdModPython import SelectTypeModifier, DeleteSelectedModifier
from ovito.io import *
import pandas as pd

import numpy as np
import seaborn as sns

# struct = LammpsData.from_file("LSF_supercell_md3.lmp", atom_style="charge").structure

# struct.to(filename="POSCAR")
# struct.remove_species(["O"])

# Load a simulation snapshot of a Cu-Zr metallic glass.
from ovito.plugins.TachyonPython import TachyonRenderer

pipeline = import_file("LSF_supercell_md3.lmp")


# The LAMMPS dump file imported above contains only numeric atom type IDs but
# no chemical element names or atom radius information. That's why we explicitly set the
# atomic radii of Cu & Zr atoms now (required for polydisperse Voronoi tessellation).
def assign_particle_radii(frame, data):
    atom_types = data.particles_.particle_types_
    # atom_types.type_by_id_(1).radius = 1.35   # Cu atomic radius assigned to atom type 1
    # atom_types.type_by_id_(2).radius = 1.55   # Zr atomic radius assigned to atom type 2


pipeline.modifiers.append(assign_particle_radii)

# Select oxygen ions and delete them
pipeline.modifiers.append(SelectTypeModifier(
    operate_on="particles",
    property="Particle Type",
    types={"O"}
))
pipeline.modifiers.append(DeleteSelectedModifier())

# Set up the Voronoi analysis modifier.
voro = VoronoiAnalysisModifier(
    compute_indices=True,
    use_radii=False,
    edge_threshold=0.0
)
pipeline.modifiers.append(voro)

# Let OVITO compute the results.
data = pipeline.compute()

# Access computed Voronoi indices.
# This is an (N) x (M) array, where M is the maximum face order.

indices = data.particles['Particle Identifier']
voro_indices = data.particles['Max Face Order']

df = pd.DataFrame(list(zip(data.particles['Particle Identifier'], data.particles['Max Face Order'], )),
                  columns=['Particle Identifier', 'Max Face Order'])

# for i in data.particles.keys():
#     df.append(data.particles(i), )
print(df)
hist_plot = sns.displot(df, x="Max Face Order")
fig = hist_plot.fig
fig.savefig("hist.pdf")

export_file(pipeline, "cations.dump", "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "Particle Type",
                     "Max Face Order"])

pipeline.add_to_scene()
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (-100, -150, 150)
vp.camera_dir = (2, 3, -3)
vp.fov = math.radians(60.0)

tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1)
vp.render_image(size='(1280, 960)', filename="figure.png", background=(1, 1, 1), alpha=False, renderer=tachyon, crop=False)

import sys
import numpy as np
import ctypes
import math
# import numpy as np
import pandas as pd
# import seaborn as sns

#  sys.path.append('C:\Users\alebon\AppData\Local\LAMMPS 64-bit 29Oct2020-MPI\bin')

#  from lammps import lammps

#  argv = sys.argv
#  if len(argv) != 2:
#      print("Syntax: simple.py in.lammps")
#      sys.exit()
#
#  infile = sys.argv[1]

#  lmp = lammps()

# TODO: sed lmp file for particle type, count ions, minimise before



from ovito.io import *
from ovito.plugins.ParticlesPython import CommonNeighborAnalysisModifier, ExpandSelectionModifier
from ovito.plugins.StdModPython import SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier

infile = "LSF_supercell_md3.lmp"


def assign_particle_types(frame, data):
    atom_types = data.particles_.particle_types_


pipeline = import_file(infile)

pipeline.modifiers.append(assign_particle_types)

pipeline.modifiers.append(SelectTypeModifier(
    operate_on="particles",
    property="Particle Type",
    types={"O"}
))
pipeline.modifiers.append(DeleteSelectedModifier())


cna = CommonNeighborAnalysisModifier(
    mode=CommonNeighborAnalysisModifier.Mode.IntervalCutoff
)
pipeline.modifiers.append(cna)


data = pipeline.compute()

df = pd.DataFrame(list(zip(data.particles['Particle Identifier'], data.particles['Structure Type'], )),
                  columns=['Particle Identifier', 'Structure Type'])

df_gb = df[df['Structure Type'] == 0]
df_bulk = df[df['Structure Type'] != 0]


del pipeline, data
pipeline2 = import_file(infile)

pipeline2.modifiers.append(assign_particle_types)


list_gb = df_gb['Particle Identifier'].values
list_bulk = df_bulk['Particle Identifier'].values


def modify(frame, data):
    #  Specify the IDs of all atoms that are to remain here
    target_ids = list_gb #  list_bulk for bulk ions
    ids = data.particles["Particle Identifier"]
    list_ids = np.in1d(ids, target_ids, assume_unique=True, invert=False)
    selection = data.particles_.create_property("Selection", data=list_ids)


pipeline2.modifiers.append(modify)

selection_mode = 'cutoff'
if selection_mode == 'cutoff':
    pipeline2.modifiers.append(ExpandSelectionModifier(cutoff=3.2,
                                                       mode=ExpandSelectionModifier.ExpansionMode.Cutoff,
                                                       iterations=1
                                                       ))
elif selection_mode == 'Nearest':
    pipeline2.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                                                       num_neighbors=1,
                                                       iterations=1
                                                       ))
else:
    print("No selection mode specified. Abort.")
    sys.exit()

pipeline2.modifiers.append(InvertSelectionModifier()) #  for bulk ions
pipeline2.modifiers.append(DeleteSelectedModifier())
data2 = pipeline2.compute()

df_2 = pd.DataFrame(list(zip(data2.particles['Particle Identifier'], data2.particles['Particle Type'], )),
                    columns=['Particle Identifier', 'Particle Type'])

df_oxygen = df_2[df_2['Particle Type'] == 4]
print("Number of oxygen:", len(df_oxygen))
df_cations = df_2[df_2['Particle Type'] != 4]
print("Number of oxygen:", len(df_cations))
percentage_oxygen = len(df_oxygen)/(len(df_oxygen)+len(df_cations))
print("Percentage of oxygen:", percentage_oxygen*100, '%')

export_file(pipeline2, "gb_temp.lmp", "lammps/data", atom_style="charge")
           # columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "Particle Type"])
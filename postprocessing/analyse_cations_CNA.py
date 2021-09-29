#!/usr/bin/env python
# coding: utf-8

# # Detection of graon boundary cations
#
# First, import classes and objects you need later.

# In[48]:


import math
# import numpy as np
import pandas as pd
# import seaborn as sns

from ovito.io import *
from ovito.plugins.ParticlesPython import CommonNeighborAnalysisModifier
from ovito.plugins.StdModPython import SelectTypeModifier, DeleteSelectedModifier


# Read the md LAMMPS data from the data file. It might be necessary to define the Particle Types in the header of the file.

# In[ ]:


pipeline = import_file("../LSF_supercell_min.lmp")


# Add modifiers to the ovito pipeline. Assign particle types, remove anions, calculate Voronoi indices.

# In[ ]:


def assign_particle_types(frame, data):
    atom_types = data.particles_.particle_types_
    # atom_types.type_by_id_(1).radius = 1.35   # Assing r to atom 1. Needed for polydisperse Voronio tess.


pipeline.modifiers.append(assign_particle_types)

# Select oxygen ions and delete them
pipeline.modifiers.append(SelectTypeModifier(
    operate_on="particles",
    property="Particle Type",
    types={"O"}
))

pipeline.modifiers.append(DeleteSelectedModifier())

# Set up the Voronoi analysis modifier.
cna = CommonNeighborAnalysisModifier(
    mode=CommonNeighborAnalysisModifier.Mode.IntervalCutoff
)
pipeline.modifiers.append(cna)


# Compute the results in ovito.

# In[ ]:


data = pipeline.compute()


# Plot a histogram for the Max face order.

# In[ ]:


df = pd.DataFrame(list(zip(data.particles['Particle Identifier'], data.particles['Structure Type'], )),
                  columns=['Particle Identifier', 'Structure Type'])

# hist_plot = sns.displot(df, x="Max Face Order", discrete=True)
# fig = hist_plot.fig
# fig.savefig("hist.pdf")


# Export cations into dump file.

# In[ ]:


export_file(pipeline, "cations.dump", "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "Particle Type",
                     "Structure Type"])


# Export two lists, one for GB cation IDs and one for bulk cation IDs.

# In[ ]:


df_gb = df[df['Structure Type'] == 0]
df_gb.to_csv('IDs_gb.csv')

df_gb = df[df['Structure Type'] != 0]
df_gb.to_csv('IDs_bulk.csv')




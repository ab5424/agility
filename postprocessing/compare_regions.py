#!/usr/bin/env python
# coding: utf-8

# Import stuff.

# In[9]:


import math
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import Image

from ovito.io import *
from ovito.plugins.ParticlesPython import VoronoiAnalysisModifier, ExpandSelectionModifier
from ovito.plugins.PyScript import Viewport
from ovito.plugins.TachyonPython import TachyonRenderer
from ovito.plugins.StdModPython import SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier


# Load structure.

# In[10]:


pipeline = import_file("../LSF_supercell_min.lmp")
def assign_particle_types(frame, data):
    atom_types = data.particles_.particle_types_
    # atom_types.type_by_id_(1).radius = 1.35   # Assing r to atom 1. Needed for polydisperse Voronio tess.


pipeline.modifiers.append(assign_particle_types)


# Load both lists.

# In[11]:


df_gb = pd.read_csv('IDs_gb.csv')
list_gb = df_gb['Particle Identifier'].values

df_bulk = pd.read_csv('IDs_bulk.csv')
list_bulk = df_bulk['Particle Identifier'].values


# Select cations by ID.

# In[12]:


def modify(frame, data):
    #Specify the IDs of all atoms that are to remain here
    target_ids = list_gb #  list_bulk for bulk ions
    ids = data.particles["Particle Identifier"]
    list_ids = np.in1d(ids, target_ids, assume_unique=True, invert=False)
    selection = data.particles_.create_property("Selection", data=list_ids)


pipeline.modifiers.append(modify)
pipeline.modifiers.append(ExpandSelectionModifier(cutoff=3.2,
                                                  mode=ExpandSelectionModifier.ExpansionMode.Cutoff,
                                                  iterations=1,
                                                  #mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                                                  #num_neighbors=2
                                                  ))
pipeline.modifiers.append(InvertSelectionModifier()) #  for bulk ions
pipeline.modifiers.append(DeleteSelectedModifier())
data = pipeline.compute()

export_file(pipeline, "gb.lmp", "lammps/data", atom_style="charge")
           # columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "Particle Type"])




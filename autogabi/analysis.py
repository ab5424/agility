"""
Analysis functions
"""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski
import sys

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.constants import codata


class GBStructure:
    """
    This class The :py:class:`polypy.read.Trajectory` class evaluates the positions
    of all atoms in the simulation.
    Args:
        backend (:py:class:`str`): List of unique atom names in trajectory.
        filename (:py:attr:`str`): Datatype of the original dataset
        e.g. DL_POLY HISTORY or CONFIG.
    """

    def __init__(self, backend, filename):
        self.backend = backend
        self.filename = filename
        self.data = None

        if self.backend not in ['ovito', 'pymatgen', 'babel', 'pyiron', 'ase']:
            # Put error here
            pass

        if self.backend == 'ovito':
            from ovito.io import import_file
            self.pipeline = import_file(str(filename))

    def delete_particles(self, particle_type):
        """
        Deletes a specific type of particles from a structure. This can be particularly useful if there is a mobile type
        in the structure. Note that for ovito structures you need to make sure that type information is included.
        Args:
            particle_type (:py:class:`str`): Timestep of desired CONFIG.
        Returns:
            config_trajectory (:py:class:`polypy.read.Trajectory`):
            Trajectory object for desired CONFIG.
        """
        if self.backend == "ovito":
            from ovito.plugins.StdModPython import SelectTypeModifier, DeleteSelectedModifier

            def assign_particle_types(frame, data):
                atom_types = data.particles_.particle_types_

            self.pipeline.modifiers.append(assign_particle_types)

            # Select oxygen ions and delete them
            self.pipeline.modifiers.append(SelectTypeModifier(
                operate_on="particles",
                property="Particle Type",
                types={particle_type}
            ))

            self.pipeline.modifiers.append(DeleteSelectedModifier())

        elif self.backend == 'pymatgen':
            pass
        elif self.backend == 'babel':
            pass

    def select_particles(self, list_ids, invert=False, delete=True, expand=True, expand_cutoff=3.2, neighbors=None,
                         iterations=1):
        """
        Selects particles by ID
        Args:
            delete:
            iterations:
            neighbors:
            expand_cutoff:
            expand:
            invert:
            list_ids:

        Returns:

        """
        if self.backend == 'ovito':
            def modify(frame, data):
                # Specify the IDs of all atoms that are to remain here
                ids = data.particles["Particle Identifier"]
                l_ids = np.in1d(ids, list_ids, assume_unique=True, invert=False)
                selection = data.particles_.create_property("Selection", data=l_ids)

            self.pipeline.modifiers.append(modify)

            if expand:
                from ovito.plugins.ParticlesPython import ExpandSelectionModifier
                if neighbors:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                                                num_neighbors=neighbors,
                                                iterations=iterations))
                else:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(cutoff=expand_cutoff,
                                                mode=ExpandSelectionModifier.ExpansionMode.Cutoff,
                                                iterations=iterations))
            if invert:
                self._invert_selection()  # for bulk ions
            if delete:
                self._delete_delection()

    def _invert_selection(self, list_ids=None):

        if self.backend == 'ovito':
            from ovito.plugins.StdModPython import InvertSelectionModifier
            self.pipeline.modifiers.append(InvertSelectionModifier())
        if self.backend == 'pymatgen':
            # Todo: Look which ids are in the list and invert by self.structure
            pass

    def _delete_delection(self):

        if self.backend == 'ovito':
            from ovito.plugins.StdModPython import DeleteSelectedModifier
            self.pipeline.modifiers.append(DeleteSelectedModifier())

    def perform_cna(self, mode='IntervalCutoff', cutoff=3.2):
        """
        Performs Common neighbor analysis.
        Returns:

        """

        if self.backend == 'ovito':
            from ovito.plugins.ParticlesPython import CommonNeighborAnalysisModifier
            if mode == 'IntervalCutoff':
                m = CommonNeighborAnalysisModifier.Mode.IntervalCutoff
            elif mode == 'AdaptiveCutoff':
                m = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
            elif mode == 'FixedCutoff':
                m = CommonNeighborAnalysisModifier.Mode.FixedCutoff
            elif mode == 'BondBased':
                m = CommonNeighborAnalysisModifier.Mode.BondBased
            else:
                print('Selected CNA Mode unknown.')
                sys.exit(1)
            cna = CommonNeighborAnalysisModifier(mode=m, cutoff=cutoff)
            self.pipeline.modifiers.append(cna)

    def perform_voroni_analysis(self):
        """
        Performs Voronoi analysis.
        Returns:

        """

        if self.backend == 'ovito':
            from ovito.plugins.ParticlesPython import VoronoiAnalysisModifier
            voro = VoronoiAnalysisModifier(
                compute_indices=True,
                use_radii=False,
                edge_threshold=0.0
            )
            self.pipeline.modifiers.append(voro)

        # https://tess.readthedocs.io/en/stable/
        # https://github.com/materialsproject/pymatgen/blob/v2022.0.14/pymatgen/analysis/structure_analyzer.py#L61-L174

    def set_analysis(self):

        if self.backend == 'ovito':
            self.data = self.pipeline.compute()

    def get_bulk_ions(self):

        if self.backend == 'ovito':
            if 'Structure Type' in self.data.particles.keys():
                df = pd.DataFrame(list(zip(self.data.particles['Particle Identifier'],
                                           self.data.particles['Structure Type'], )),
                                  columns=['Particle Identifier', 'Structure Type'])
                df_gb = df[df['Structure Type'] == 0]
                return list(df_gb['Particle Identifier'])

    # Todo: Verkippungswinkel
    # Todo: Grain Index


class GBStructureTimeseries(GBStructure):
    """
    This is a class containing multiple snapshots from a time series.
    """

    def remove_timesteps(self, timesteps_to_exclude):
        """
        Removes timesteps from the beggining of a simulation
        Args:
            timesteps_to_exclude (:py:class:`int`): Number of timesteps to exclude
        Returns:
            new_trajectory (:py:class:`polypy.read.Trajectory`):
            Trajectory object.
        """
        pass

    # Todo: Add differentiation between diffusion along a grain boundary, transverse to the GB, and between grains

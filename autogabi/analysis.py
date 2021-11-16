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
    This is the fundamental class of a grain boundary object.
    """

    def __init__(self, backend, filename):
        self.backend = backend
        self.filename = filename
        self.data = None

        if self.backend not in [
            "ovito",
            "pymatgen",
            "babel",
            "pyiron",
            "ase",
            "lammps",
        ]:
            # Put error here
            pass

        if self.backend == "ovito":
            from ovito.io import import_file

            self.pipeline = import_file(str(filename))

        if self.backend == "pymatgen":
            from pymatgen.core import Structure

            self.data.structure = Structure.from_file(filename)

    def delete_particles(self, particle_type):
        """
        Deletes a specific type of particles from a structure. This can be particularly useful if there is a mobile type
        in the structure. Note that for ovito structures you need to make sure that type information is included.
        Args:
            particle_type (:py:class:`str`):
        Returns:
        """
        if self.backend == "ovito":
            from ovito.plugins.StdModPython import (
                SelectTypeModifier,
                DeleteSelectedModifier,
            )

            def assign_particle_types(frame, data):
                atom_types = data.particles_.particle_types_

            self.pipeline.modifiers.append(assign_particle_types)

            # Select atoms and delete them
            self.pipeline.modifiers.append(
                SelectTypeModifier(
                    operate_on="particles",
                    property="Particle Type",
                    types={particle_type},
                )
            )

            self.pipeline.modifiers.append(DeleteSelectedModifier())

        elif self.backend == "pymatgen":
            self.data.structure.remove_species(particle_type)

        elif self.backend == "babel":
            pass

    def select_particles(
        self,
        list_ids,
        invert=True,
        delete=True,
        expand=False,
        expand_cutoff=3.2,
        nearest_neighbors=None,
        iterations=1,
    ):
        """
        Selects particles by ID
        Args:
            nearest_neighbors:
            delete:
            iterations:
            expand_cutoff:
            expand:
            invert:
            list_ids:

        Returns:

        """
        if self.backend == "ovito":

            def modify(frame, data):
                # Specify the IDs of all atoms that are to remain here
                ids = data.particles["Particle Identifier"]
                l_ids = np.in1d(ids, list_ids, assume_unique=True, invert=False)
                selection = data.particles_.create_property("Selection", data=l_ids)

            self.pipeline.modifiers.append(modify)

            if expand:
                from ovito.plugins.ParticlesPython import ExpandSelectionModifier

                if nearest_neighbors:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(
                            mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                            num_neighbors=nearest_neighbors,
                            iterations=iterations,
                        )
                    )
                else:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(
                            cutoff=expand_cutoff,
                            mode=ExpandSelectionModifier.ExpansionMode.Cutoff,
                            iterations=iterations,
                        )
                    )
            if invert:
                self._invert_selection()  # for bulk ions
            if delete:
                self._delete_delection()

    def _invert_selection(self, list_ids=None):

        if self.backend == "ovito":
            from ovito.plugins.StdModPython import InvertSelectionModifier

            self.pipeline.modifiers.append(InvertSelectionModifier())

        if self.backend == "pymatgen":
            # Todo: Look which ids are in the list and invert by self.structure
            pass

    def _delete_delection(self):

        if self.backend == "ovito":
            from ovito.plugins.StdModPython import DeleteSelectedModifier

            self.pipeline.modifiers.append(DeleteSelectedModifier())

    def perform_cna(self, mode="IntervalCutoff", cutoff=3.2):
        """
        Performs Common neighbor analysis.
        Returns:

        """

        if self.backend == "ovito":
            from ovito.plugins.ParticlesPython import CommonNeighborAnalysisModifier

            if mode == "IntervalCutoff":
                m = CommonNeighborAnalysisModifier.Mode.IntervalCutoff
            elif mode == "AdaptiveCutoff":
                m = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
            elif mode == "FixedCutoff":
                m = CommonNeighborAnalysisModifier.Mode.FixedCutoff
            elif mode == "BondBased":
                m = CommonNeighborAnalysisModifier.Mode.BondBased
            else:
                print("Selected CNA Mode unknown.")
                sys.exit(1)
            cna = CommonNeighborAnalysisModifier(mode=m, cutoff=cutoff)
            self.pipeline.modifiers.append(cna)

    def perform_voroni_analysis(self):
        """
        Performs Voronoi analysis.
        Returns:

        """

        if self.backend == "ovito":
            from ovito.plugins.ParticlesPython import VoronoiAnalysisModifier

            voro = VoronoiAnalysisModifier(
                compute_indices=True, use_radii=False, edge_threshold=0.0
            )
            self.pipeline.modifiers.append(voro)

        # https://tess.readthedocs.io/en/stable/
        # https://github.com/materialsproject/pymatgen/blob/v2022.0.14/pymatgen/analysis/structure_analyzer.py#L61-L174

    def perform_ptm(
        self,
        enabled: list = ["fcc", "hpc", "bcc"],
        compute: bool = True,
        *args,
        **kwargs
    ):
        """
        Perform Polyhedral template matching.
        https://github.com/pmla/polyhedral-template-matching

        Args:
            enabled (list): List of strings for enabled structure types. Possible values:
                fcc-hcp-bcc-ico-sc-dcub-dhex-graphene
            for ovito:
                output_deformation_gradient = False
                output_interatomic_distance = False
                output_ordering = False
                output_orientation = False
                output_rmsd = False
                rmsd_cutoff = 0.1
            for lammps:
                ID = 1
                group-ID = all
                threshold = 0.1
                group2-ID = all

        Returns:

        """

        if self.backend == "ovito":

            from ovito.plugins.ParticlesPython import PolyhedralTemplateMatchingModifier

            ptm = PolyhedralTemplateMatchingModifier(*args, **kwargs)

            # Enabled by default: FCC, HCP, BCC
            if "fcc" not in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.FCC
                ].enabled = False
            if "hcp" not in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.HCP
                ].enabled = False
            if "bcc" not in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.BCC
                ].enabled = False
            if "ico" in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.ICO
                ].enabled = True
            if "sc" in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.SC
                ].enabled = True
            if "dcub" in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND
                ].enabled = True
            if "dhex" in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND
                ].enabled = True
            if "graphene" in enabled:
                ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.GRAPHENE
                ].enabled = True

            self.pipeline.modifiers.append(ptm)

            if compute:
                self.data = self.pipeline.compute()

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_ptm_atom.html
            pass
        else:
            # print error
            pass

    def perform_ajm(self):
        # AcklandJonesModifier
        pass

    def get_distinct_grains(self, *args, **kwargs):
        """
        Get distinct grains from the structure.
        Args:
            ovito:
                    algorithm = GrainSegmentationModifier.Algorithm.GraphClusteringAuto
                    color_particles = True
                    handle_stacking_faults = True
                    merging_threshold = 0.0
                    min_grain_size = 100
                    orphan_adoption = True

        Returns:

        """

        if self.backend == "ovito":
            from ovito.plugins.CrystalAnalysisPython import GrainSegmentationModifier

            gsm = GrainSegmentationModifier(*args, **kwargs)
            self.pipeline.modifiers.append(gsm)
            self.data = self.pipeline.compute()

    def set_analysis(self):

        if self.backend == "ovito":
            self.data = self.pipeline.compute()

    def get_gb_atoms(self):

        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                df = pd.DataFrame(
                    list(
                        zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                    ),
                    columns=["Particle Identifier", "Structure Type"],
                )
                df_gb = df[df["Structure Type"] == 0]
                return list(df_gb["Particle Identifier"])

    def get_bulk_atoms(self):

        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                df = pd.DataFrame(
                    list(
                        zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                    ),
                    columns=["Particle Identifier", "Structure Type"],
                )
                df_gb = df[df["Structure Type"] != 0]
                return list(df_gb["Particle Identifier"])

    def get_type(self, atom_type):

        if self.backend == "ovito":
            # Currently doesn't work!
            # def assign_particle_types(frame, data):
            #     atom_types = data.particles_.particle_types_
            #
            # self.pipeline.modifiers.append(assign_particle_types)
            # self.set_analysis()
            df = pd.DataFrame(
                list(
                    zip(
                        self.data.particles["Particle Identifier"],
                        self.data.particles["Particle Type"],
                    )
                ),
                columns=["Particle Identifier", "Particle Type"],
            )
            df_atom = df[df["Particle Type"].eq(atom_type)]
            return list(df_atom["Particle Identifier"])

    # Todo: Verkippungswinkel
    # Todo: Grain Index

    def get_fraction(self, numerator, denominator):

        if self.backend == "ovito":
            num = sum([len(self.get_type(i)) for i in numerator])
            den = sum([len(self.get_type(i)) for i in denominator])
            return num / den


class GBStructureTimeseries(GBStructure):
    """
    This is a class containing multiple snapshots from a time series.
    """

    # Todo: get diffusion data
    # Todo: differentiate between along/across GB

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

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Analysis functions."""


import sys

import numpy as np
import pandas as pd

from agility.minimiser import mimimise_lmp


class GBStructure:
    """This is the fundamental class of a grain boundary object."""

    def __init__(self, backend, filename, **kwargs):
        """Initialize."""
        self.backend = backend
        self.filename = filename
        self.data = None

        if self.backend not in [
            "ovito",
            "pymatgen",
            "babel",
            "pyiron",
            # https://github.com/pyiron/pylammpsmpi
            "ase",
            "lammps",
        ]:
            # Put error here
            pass

        if self.backend == "lammps":
            # Determine if a jupyter notebook is used
            # Taken from shorturl.at/aikzP
            try:
                shell = get_ipython().__class__.__name__
                if shell == "ZMQInteractiveShell":
                    ipy = True  # Jupyter notebook or qtconsole
                elif shell == "TerminalInteractiveShell":
                    ipy = False  # Terminal running IPython
                else:
                    ipy = False  # Other type (?)
            except NameError:
                ipy = False  # Probably standard Python interpreter

            if ipy:
                from lammps import IPyLammps

                self.pylmp = IPyLammps()
            else:
                from lammps import PyLammps

                self.pylmp = PyLammps()

        if filename:
            self.read_file(filename, **kwargs)

    def read_file(self, filename, **kwargs):
        """Read structure from file.

        Args:
            filename: File to read.
        Returns:
            None
        """
        if self.backend == "ovito":
            from ovito.io import import_file

            self.pipeline = import_file(str(filename))

        if self.backend == "pymatgen":
            from pymatgen.core import Structure

            self.data.structure = Structure.from_file(filename)

        if self.backend == "lammps":
            self._init_lmp(filename=filename, **kwargs)

    def _init_lmp(
        self,
        filename,
        file_type: str = "data",
        pair_style: str = "none",
        kspace_style: str = "none",
    ):
        """Initialise lammps backend.

        Args:
            filename: File to read.
            file_type: File type (data, dump, restart)
            pair_style: lammps pair style
            kspace_style:
        """
        self.pylmp.units("metal")
        self.pylmp.atom_style("charge")
        self.pylmp.atom_modify("map array")
        self.pylmp.pair_style(f"{pair_style}")
        if kspace_style:
            self.pylmp.kspace_style(f"{kspace_style}")
        if file_type == "data":
            self.pylmp.read_data(filename)
        elif file_type == "dump":
            self.pylmp.read_dump(filename)
        elif file_type == "restart":
            self.pylmp.read_restart(filename)
        else:
            print("Please specify the type of lammps file to read.")

    def save_structure(self, filename: str = None, file_type: str = None, **kwargs):
        """Save structure to disc.

        Args:
            filename:
            file_type:

        """
        if self.backend == "ovito":
            from ovito.io import export_file

            export_file(self.pipeline, filename, file_type, **kwargs)

        if self.backend == "lammps":
            if file_type == "data":
                self.pylmp.write_data(filename)
            elif file_type == "dump":
                self.pylmp.write_dump(filename)
            elif file_type == "restart":
                self.pylmp.write_restart(filename)

    def minimise(self, *args, **kwargs):
        """Minimise structure.

        Returns:
        """
        if self.backend == "ovito":
            print(f"The {self.backend} backend does not support minimisation.")
            sys.exit(1)
        elif self.backend == "lammps":
            self.pylmp = mimimise_lmp(self.pylmp, *args, **kwargs)

    def delete_particles(self, particle_type):
        """Delete a specific type of particles from a structure.

        This can be particularly useful if
        there is a mobile type in the structure. Note that for ovito structures you need to make
        sure that type information is included.
        Args:
            particle_type:
        Returns:
        """
        if self.backend == "ovito":
            # from ovito.plugins.StdModPython import (
            #     SelectTypeModifier,
            #     DeleteSelectedModifier,
            # )
            from ovito.modifiers import DeleteSelectedModifier, SelectTypeModifier

            def assign_particle_types(frame, data):  # pylint: disable=W0613
                atom_types = data.particles_.particle_types_  # pylint: disable=W0612

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
        elif self.backend == "lammps":
            self.pylmp.group(f"delete type {particle_type}")
            self.pylmp.delete_atoms("group delete compress no")

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
        """Select particles by ID.

        Args:
            nearest_neighbors:
            delete:
            iterations:
            expand_cutoff:
            expand:
            invert:
            list_ids:

        Returns:
            None
        """
        if self.backend == "ovito":

            def modify(frame, data):  # pylint: disable=W0613
                # Specify the IDs of all atoms that are to remain here
                ids = data.particles["Particle Identifier"]
                l_ids = np.in1d(ids, list_ids, assume_unique=True, invert=False)
                selection = data.particles_.create_property(
                    "Selection", data=l_ids
                )  # pylint: disable=W0612

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
                self._delete_selection()

    def _invert_selection(self):

        if self.backend == "ovito":
            from ovito.plugins.StdModPython import InvertSelectionModifier

            self.pipeline.modifiers.append(InvertSelectionModifier())

        if self.backend == "pymatgen":
            # Todo: Look which ids are in the list and invert by self.structure
            pass

    def _delete_selection(self):

        if self.backend == "ovito":
            from ovito.plugins.StdModPython import DeleteSelectedModifier

            self.pipeline.modifiers.append(DeleteSelectedModifier())

    def perform_cna(self, mode: str = "IntervalCutoff", cutoff: float = 3.2, compute: bool = True):
        """Perform Common neighbor analysis.

        Args:
            mode: Mode of common neighbor analysis. The lammps backend uses "FixedCutoff".
            cutoff: Cutoff for the FixedCutoff mode.
            compute: Compute results.
        Returns:
            None
        """
        if self.backend == "ovito":
            # TODO: Enable/diable structure types
            from ovito.plugins.ParticlesPython import CommonNeighborAnalysisModifier

            if mode == "IntervalCutoff":
                cna_mode = CommonNeighborAnalysisModifier.Mode.IntervalCutoff
            elif mode == "AdaptiveCutoff":
                cna_mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
            elif mode == "FixedCutoff":
                cna_mode = CommonNeighborAnalysisModifier.Mode.FixedCutoff
            elif mode == "BondBased":
                cna_mode = CommonNeighborAnalysisModifier.Mode.BondBased
            else:
                print(f'Selected CNA mode "{mode}" unknown.')
                sys.exit(1)
            cna = CommonNeighborAnalysisModifier(mode=cna_mode, cutoff=cutoff)
            self.pipeline.modifiers.append(cna)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_cna_atom.html
            n_compute = len([i["style"] for i in self.pylmp.computes if i["style"] == "cna/atom"])
            self.pylmp.compute(f"cna_{n_compute} all cna/atom {cutoff}")

        else:
            raise NotImplementedError(f"The backend {self.backend} doesn't support this function.")

        if compute:
            self.set_analysis()

    def perfom_cnp(self, cutoff: float = 3.20, compute: bool = False):
        """Perform Common Neighborhood Parameter calculation.

        Returns:
            None
        """
        if self.backend == "lammps":
            self.pylmp.compute(f"compute 1 all cnp/atom {cutoff}")

        if compute:
            self.set_analysis()

    def perform_voroni_analysis(self, compute: bool = False):
        """Perform Voronoi analysis.

        Args:
            ovito:
                bonds_vis = False
                edge_threshold = 0.0
                face_threshold = 0.0
                generate_bonds = False
                generate_polyhedra = False
                mesh_vis = False
                relative_face_threshold = 0.0
                use_radii = False
            lammps:
                only_group = no arg
                occupation = no arg
                surface arg = sgroup-ID
                  sgroup-ID = compute the dividing surface between group-ID and sgroup-ID
                    this keyword adds a third column to the compute output
                radius arg = v_r
                  v_r = radius atom style variable for a poly-disperse Voronoi tessellation
                edge_histo arg = maxedge
                  maxedge = maximum number of Voronoi cell edges to be accounted in the histogram
                edge_threshold arg = minlength
                  minlength = minimum length for an edge to be counted
                face_threshold arg = minarea
                  minarea = minimum area for a face to be counted
                neighbors value = yes or no = store list of all neighbors or no
                peratom value = yes or no = per-atom quantities accessible or no
        Returns:
            None
        """
        if self.backend == "ovito":
            from ovito.plugins.ParticlesPython import VoronoiAnalysisModifier

            voro = VoronoiAnalysisModifier(
                compute_indices=True, use_radii=False, edge_threshold=0.0
            )
            self.pipeline.modifiers.append(voro)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_voronoi_atom.html
            self.pylmp.compute("1 all voronoi/atom")

        if compute:
            self.set_analysis()

        # https://tess.readthedocs.io/en/stable/
        # https://github.com/materialsproject/pymatgen/blob/v2022.0.14/pymatgen/analysis/structure_analyzer.py#L61-L174

    def perform_ptm(
        self,
        *args,
        enabled: list = ["fcc", "hpc", "bcc"],
        rmsd_threshold: float = 0.1,
        compute: bool = True,
        **kwargs,
    ):
        """Perform Polyhedral template matching.

        https://dx.doi.org/10.1088/0965-0393/24/5/055007
        https://github.com/pmla/polyhedral-template-matching.

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
            None
        """
        for i in enabled:
            if i not in ["fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene"]:
                print(f"Enabled structure type {i} unknown")
        if self.backend == "ovito":

            from ovito.plugins.ParticlesPython import PolyhedralTemplateMatchingModifier

            ptm = PolyhedralTemplateMatchingModifier(*args, rmsd_cutoff=rmsd_threshold, **kwargs)

            # Enabled by default: FCC, HCP, BCC
            if "fcc" not in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = False
            if "hcp" not in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = False
            if "bcc" not in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = False
            if "ico" in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
            if "sc" in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
            if "dcub" in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
            if "dhex" in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
            if "graphene" in enabled:
                ptm.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = True

            self.pipeline.modifiers.append(ptm)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_ptm_atom.html
            n_compute = len([i["style"] for i in self.pylmp.computes if i["style"] == "ptm/atom"])
            enabled_structures = " ".join(enabled)
            self.pylmp.compute(
                f"ptm_{n_compute} all ptm/atom {enabled_structures} {rmsd_threshold}"
            )
        else:
            # print error
            pass

        if compute:
            self.set_analysis()

    def perform_ajm(self, compute: bool = True):
        """Ackland-Jones analysis.

        https://doi.org/10.1103/PhysRevB.73.054104
        Returns:
        """
        if self.backend == "ovito":
            from ovito.plugins.ParticlesPython import AcklandJonesModifier

            ajm = AcklandJonesModifier()
            self.pipeline.modifiers.append(ajm)

            if compute:
                self.data = self.pipeline.compute()

        elif self.backend == "lammps":
            n_compute = len(
                [i["style"] for i in self.pylmp.computes if i["style"] == "ackland/atom"]
            )
            self.pylmp.compute(f"ackland_{n_compute} all ackland/atom")

        else:
            pass

        if compute:
            self.set_analysis()

    def perform_csp(self, num_neighbors: int = 12, compute: bool = True):
        """Centrosymmetric parameter.

        Use 12 for fcc and 8 for bcc, respectively
        Returns:
        """
        if self.backend == "ovito":
            from ovito.plugins.ParticlesPython import CentroSymmetryModifier

            csp = CentroSymmetryModifier()
            self.pipeline.modifiers.append(csp)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_centro_atom.html
            n_compute = len(
                [i["style"] for i in self.pylmp.computes if i["style"] == "centro/atom"]
            )
            self.pylmp.compute(f"centro_{n_compute} all centro/atom {num_neighbors}")

        if compute:
            self.set_analysis()

    def get_distinct_grains(
        self, *args, algorithm: str = "GraphClusteringAuto", compute: bool = True, **kwargs
    ):
        """Get distinct grains from the structure.

        Args:
            ovito:
                    algorithm = GrainSegmentationModifier.Algorithm.GraphClusteringAuto
                    color_particles = True
                    handle_stacking_faults = True
                    merging_threshold = 0.0
                    min_grain_size = 100
                    orphan_adoption = True

        Returns:
            None
        """
        if self.backend == "ovito":
            from ovito.plugins.CrystalAnalysisPython import GrainSegmentationModifier

            if algorithm == "GraphClusteringAuto":
                gsm_mode = GrainSegmentationModifier.Algorithm.GraphClusteringAuto
            elif algorithm == "GraphClusteringManual":
                gsm_mode = GrainSegmentationModifier.Algorithm.GraphClusteringManual
            elif algorithm == "MinimumSpanningTree":
                gsm_mode = GrainSegmentationModifier.Algorithm.MinimumSpanningTree
            else:
                print("Incorrenct Grain Segmentation algorithm specified.")
                sys.exit(1)

            gsm = GrainSegmentationModifier(*args, algorithm=gsm_mode, **kwargs)
            self.pipeline.modifiers.append(gsm)
            if compute:
                self.data = self.pipeline.compute()
            # TODO: Get misorientation plot

    def set_analysis(self):
        """Compute results.

        Important function for the ovito backend. The lammps backend can access compute results
        without evaluation of this function.
        Returns:
            None
        """
        if self.backend == "ovito":
            self.data = self.pipeline.compute()

        elif self.backend == "lammps":
            self.pylmp.run(1)

        elif self.backend == "pymatgen":
            print("The pymatgen backend does not require setting the analysis.")

    def get_gb_atoms(self, mode: str = "cna"):
        """Get the atoms at the grain boundary.

        For this to work, some sort of stuctural analysis has to be performed.

        Args:
            mode: Mode for selection of grain boundary atoms.
        Returns:
            None
        """
        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                df_temp = pd.DataFrame(
                    list(
                        zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                    ),
                    columns=["Particle Identifier", "Structure Type"],
                )
                df_gb = df_temp[df_temp["Structure Type"] == 0]
                return list(df_gb["Particle Identifier"])
            else:
                print("No Structure analysis performed.")
                sys.exit(1)
        elif self.backend == "lammps":
            # Supported analysis methods: cna, ptm,
            from lammps import LMP_STYLE_ATOM, LMP_TYPE_VECTOR

            # ids = []
            # for i in range(len(self.pylmp.atoms)):
            #     ids.append(self.pylmp.atoms[i].id)
            types = np.concatenate(
                self.pylmp.lmp.numpy.extract_compute("cna_0", LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            )
            # https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc
            ids = np.concatenate(self.pylmp.lmp.numpy.extract_atom("id"))
            df_temp = pd.DataFrame(
                list(
                    zip(
                        ids,
                        types,
                    )
                ),
                columns=["Particle Identifier", "Structure Type"],
            )
            # TDOD: This is only cna, what about others?
            if mode == "cna":
                df_gb = df_temp[df_temp["Structure Type"] == 5]
            elif mode == "ptm" or mode == "ackland":
                df_gb = df_temp[df_temp["Structure Type"] == 0]
            elif mode == "voronoi" or mode == "centro":
                print("Method not implemented.")
                sys.exit(1)
            return list(df_gb["Particle Identifier"])
        else:
            print("Method not implemented.")
            return None

    def get_bulk_atoms(self):
        """Get the atoms in the bulk, as determined by structural analysis.

        Returns:
            None
        """
        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                df_temp = pd.DataFrame(
                    list(
                        zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                    ),
                    columns=["Particle Identifier", "Structure Type"],
                )
                df_gb = df_temp[df_temp["Structure Type"] != 0]
                return list(df_gb["Particle Identifier"])
        elif self.backend == "lammps":
            # TODO
            return None
        else:
            print("Method not implemented.")
            return None

    def get_type(self, atom_type):
        """Get all atoms by type.

        Args:
            atom_type:

        Returns:
            None
        """
        if self.backend == "ovito":
            # Currently doesn't work!
            # def assign_particle_types(frame, data):
            #     atom_types = data.particles_.particle_types_
            #
            # self.pipeline.modifiers.append(assign_particle_types)
            # self.set_analysis()
            df_temp = pd.DataFrame(
                list(
                    zip(
                        self.data.particles["Particle Identifier"],
                        self.data.particles["Particle Type"],
                    )
                ),
                columns=["Particle Identifier", "Particle Type"],
            )
            df_atom = df_temp[df_temp["Particle Type"].eq(atom_type)]
            return list(df_atom["Particle Identifier"])

        elif self.backend == "lammps":
            # TODO
            return None
        else:
            print("Method not implemented.")
            return None

    # Todo: Verkippungswinkel
    # Todo: Grain Index

    def get_fraction(self, numerator, denominator):
        """Get fraction of ions/atoms. Helper function.

        Args:
            numerator:
            denominator:

        Returns:
            None
        """
        if self.backend == "ovito":
            num = sum([len(self.get_type(i)) for i in numerator])
            den = sum([len(self.get_type(i)) for i in denominator])
            return num / den

        else:
            print("Method not implemented.")
            return None

    def save_image(self, filename: str = "image.png"):
        """Save image file.

        Args:
            filename: file to be saved.

        """
        if self.backend == "ovito":
            # TODO: use render function
            pass
        if self.backend == "lammps":
            # Only works with IPython integration
            self.pylmp.image(filename=filename)

    def convert_backend(self, convert_to: str = None):
        """Convert the current backend.

        Args:
            convert_to: Backend to convert to.
        """
        if self.backend == "lammps":
            from datetime import datetime

            filename = datetime.now().strftime("%d%m%Y_%H%M%S") + ".lmp"
            self.save_structure("filename", file_type="data")
            if convert_to == "ovito":
                import pathlib
                from ovito.io import import_file

                self.backend == convert_to
                self.pipeline = import_file(str(filename))
                del self.pylmp
                tempfile = pathlib.Path(filename)
                tempfile.unlink()


class GBStructureTimeseries(GBStructure):
    """This is a class containing multiple snapshots from a time series."""

    # Todo: get diffusion data
    # Todo: differentiate between along/across GB

    def remove_timesteps(self, timesteps_to_exclude):
        """Remove timesteps from the beggining of a simulation.

        Args:
            timesteps_to_exclude (:py:class:`int`): Number of timesteps to exclude
        Returns:
            new_trajectory (:py:class:`polypy.read.Trajectory`):
            Trajectory object.
        """
        pass

    # Todo: Add differentiation between diffusion along a grain boundary, transverse to the GB,
    # and between grains

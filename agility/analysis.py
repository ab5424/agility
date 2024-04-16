# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

"""Analysis functions."""

from __future__ import annotations

import pathlib
import random
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from agility.minimiser import mimimise_lmp

available_backends = Literal["ovito", "pymatgen", "babel", "pyiron", "ase", "lammps"]
# https://github.com/pyiron/pylammpsmpi


class GBStructure:
    """Fundamental class of a grain boundary object."""

    def __init__(
        self,
        backend: available_backends,
        filename: str | pathlib.Path,
        **kwargs,
    ) -> None:
        """Initialize."""
        self.backend = backend
        self.filename = filename
        self.data: Any = None

        if self.backend == "lammps":
            # Determine if a jupyter notebook is used
            # Taken from shorturl.at/aikzP
            try:
                shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
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

    def read_file(self, filename: str | pathlib.Path, **kwargs) -> None:
        """Read structure from file.

        Args:
            filename: File to read.
            **kwargs: Additional arguments for reading the file.

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
        filename: str | pathlib.Path,
        file_type: str = "data",
        pair_style: str = "none",
        kspace_style: str = "none",
    ) -> None:
        """Initialise lammps backend.

        Args:
            filename: File to read.
            file_type (str): File type (data, dump, restart)
            pair_style (str): lammps pair style
            kspace_style (str): lammps kspace style
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
            msg = "Please specify the type of lammps file to read."
            raise ValueError(msg)

    def save_structure(self, filename: str, file_type: str, **kwargs) -> None:
        """Save structure to disc.

        Args:
            filename (str): Filename to save.
            file_type (str): File type (data, dump, restart)
            **kwargs: Additional arguments for saving the file.
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

    def minimise(self, *args, **kwargs) -> None:
        """Minimise structure."""
        if self.backend == "ovito":
            msg = f"The {self.backend} backend has no minimisation capabilities."
            raise NotImplementedError(msg)
        if self.backend == "lammps":
            self.pylmp = mimimise_lmp(self.pylmp, *args, **kwargs)
        else:
            msg = f"The {self.backend} backend does not support minimisation yet."
            raise NotImplementedError(msg)

    def delete_particles(self, particle_type: set) -> None:
        """Delete a specific type of particles from a structure.

        This can be particularly useful if
        there is a mobile type in the structure. Note that for ovito structures you need to make
        sure that type information is included.

        Args:
            particle_type: Particle type to delete.

        """
        if self.backend == "ovito":
            from ovito.modifiers import DeleteSelectedModifier

            self.select_particles_by_type(particle_type)
            self.pipeline.modifiers.append(DeleteSelectedModifier())

        elif self.backend == "lammps":
            self.pylmp.group(f"delete type {particle_type}")
            self.pylmp.delete_atoms("group delete compress no")

        elif self.backend == "pymatgen":
            self.data.structure.remove_species(particle_type)

        elif self.backend == "babel":
            pass

    #    def assign_particles_types(self, particle_types: list):
    #        """Assign
    #
    #        """
    #        if self.backend == "ovito":
    #            def setup_atom_types(frame, data):
    #                types = self.data.particles_.particle_types_
    #                for i, particle_type in enumerate(particle_types):
    #                    types.type_by_id_(i+1).name = particle_type
    #            self.pipeline.modifiers.append(setup_atom_types)
    #            self.set_analysis()

    def select_particles_by_type(self, particle_type: set) -> None:
        """Select a specific type of particles from a structure.

        Args:
            particle_type (set): Particle type to select.
        """
        if self.backend == "ovito":
            # from ovito.plugins.StdModPython import (
            #     SelectTypeModifier,
            #     DeleteSelectedModifier,
            # )
            from ovito.modifiers import SelectTypeModifier

            def assign_particle_types(  # noqa: ANN202
                frame,  # noqa: ANN001,ARG001
                data,  # noqa: ANN001
            ):  # pylint: disable=W0613
                atom_types = data.particles_.particle_types_  # pylint: disable=W0612

            self.pipeline.modifiers.append(assign_particle_types)

            # Select atoms and delete them
            self.pipeline.modifiers.append(
                SelectTypeModifier(  # type: ignore[call-arg]
                    operate_on="particles",
                    property="Particle Type",
                    types=particle_type,
                ),
            )

    def select_particles(
        self,
        list_ids: list,
        list_ids_type: Literal["Identifier", "Indices"] = "Identifier",
        invert: bool = True,
        delete: bool = True,
        expand_cutoff: float | None = None,
        expand_nearest_neighbors: int | None = None,
        iterations: int = 1,
    ) -> None:
        """Select particles by ID.

        Args:
            list_ids (list): List of IDs to select.
            list_ids_type (str): "Indices" or "Identifier"
            expand_nearest_neighbors (int): Number of nearest neighbors. Default 1.
            invert (bool): Invert selection.
            delete (bool): Delete selection.
            iterations (int): Number of iterations for expansion.
            expand_cutoff (float): Expansion cutoff. Default 3.2.

        Returns:
            None
        """
        if self.backend == "ovito":
            if np.where(self.data.particles.selection != 0)[0].size > 0:
                self._clear_selection()
                warnings.warn("Selection currently not empty. Clearing selection.", stacklevel=2)

            def modify(frame, data):  # noqa: ANN001,ANN202,ARG001  # pylint: disable=W0613
                # Specify the IDs of all atoms that are to remain here
                if list_ids_type == "Identifier":
                    ids = data.particles["Particle Identifier"]
                elif list_ids_type == "Indices":
                    ids = list(np.where(self.data.particles["Structure Type"] != 10000)[0])
                else:
                    msg = "Only Indices and Identifier are possible as list id types."
                    raise NameError(msg)
                l_ids = np.in1d(ids, list_ids, assume_unique=True, invert=False)
                selection = data.particles_.create_property(  # pylint: disable=W0612
                    "Selection",
                    data=l_ids,
                )

            self.pipeline.modifiers.append(modify)

            if expand_nearest_neighbors or expand_cutoff:
                from ovito.plugins.ParticlesPython import ExpandSelectionModifier

                if expand_nearest_neighbors:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(
                            mode=ExpandSelectionModifier.ExpansionMode.Nearest,
                            num_neighbors=expand_nearest_neighbors,
                            iterations=iterations,
                        ),
                    )
                else:
                    self.pipeline.modifiers.append(
                        ExpandSelectionModifier(
                            cutoff=expand_cutoff,
                            mode=ExpandSelectionModifier.ExpansionMode.Cutoff,
                            iterations=iterations,
                        ),
                    )
            if invert:
                self._invert_selection()  # for bulk ions
            if delete:
                self._delete_selection()

    def _invert_selection(self) -> None:
        if self.backend == "ovito":
            from ovito.modifiers import InvertSelectionModifier

            self.pipeline.modifiers.append(InvertSelectionModifier())

        if self.backend == "pymatgen":
            # TODO: Look which ids are in the list and invert by self.structure
            pass

    def _delete_selection(self) -> None:
        if self.backend == "ovito":
            from ovito.modifiers import DeleteSelectedModifier

            self.pipeline.modifiers.append(DeleteSelectedModifier())

    def _clear_selection(self) -> None:
        if self.backend == "ovito":
            from ovito.modifiers import ClearSelectionModifier

            self.pipeline.modifiers.append(ClearSelectionModifier())

    def perform_cna(
        self,
        mode: str = "IntervalCutoff",
        enabled: Sequence[str] = ("fcc", "hpc", "bcc"),
        cutoff: float = 3.2,
        color_by_type: bool = True,
        only_selected: bool = False,
        compute: bool = True,
    ) -> None:
        """Perform Common neighbor analysis.

        Args:
            mode: Mode of common neighbor analysis. The lammps backend uses "FixedCutoff".
            enabled: Enabled structures for identifier.
            cutoff: Cutoff for the FixedCutoff mode.
            color_by_type: Color by structure type.
            only_selected: Only selected particles.
            compute: Compute results.

        Returns:
            None
        """
        if self.backend == "ovito":
            # TODO: Enable/disable structure types
            from ovito.modifiers import CommonNeighborAnalysisModifier

            cna_modes = {
                "IntervalCutoff": CommonNeighborAnalysisModifier.Mode.IntervalCutoff,
                "AdaptiveCutoff": CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff,
                "FixedCutoff": CommonNeighborAnalysisModifier.Mode.FixedCutoff,
                "BondBased": CommonNeighborAnalysisModifier.Mode.BondBased,
            }
            if mode in cna_modes:
                cna_mode = cna_modes[mode]
            else:
                msg = f'Selected CNA mode "{mode}" unknown.'
                raise ValueError(msg)

            _cna = CommonNeighborAnalysisModifier(  # type: ignore[call-arg]
                mode=cna_mode,
                cutoff=cutoff,
                color_by_type=color_by_type,
                only_selected=only_selected,
            )
            # Enabled by default: FCC, HCP, BCC
            if "fcc" not in enabled:
                _cna.structures[
                    CommonNeighborAnalysisModifier.Type.FCC  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "hcp" not in enabled:
                _cna.structures[
                    CommonNeighborAnalysisModifier.Type.HCP  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "bcc" not in enabled:
                _cna.structures[
                    CommonNeighborAnalysisModifier.Type.BCC  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "ico" not in enabled:
                _cna.structures[
                    CommonNeighborAnalysisModifier.Type.ICO  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]

            self.pipeline.modifiers.append(_cna)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_cna_atom.html
            n_compute = len([i["style"] for i in self.pylmp.computes if i["style"] == "cna/atom"])
            self.pylmp.compute(f"cna_{n_compute} all cna/atom {cutoff}")

        else:
            raise not_implemented(self.backend)

        if only_selected:
            warnings.warn(
                "Evaluating only the selected atoms. Be aware that non-selected atoms may be "
                "assigned to the wrong category.",
                stacklevel=2,
            )
        if compute:
            self.set_analysis()

    def perform_cnp(self, cutoff: float = 3.20, compute: bool = False) -> None:
        """Perform Common Neighborhood Parameter calculation.

        Please cite https://doi.org/10.1016/j.cpc.2007.05.018

        Returns:
            None
        """
        if self.backend == "lammps":
            self.pylmp.compute(f"compute 1 all cnp/atom {cutoff}")

        if compute:
            self.set_analysis()

    def perform_voronoi_analysis(self, compute: bool = False) -> None:
        """Perform Voronoi analysis.

        Args:
            compute (bool): Compute results.
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
                compute_indices=True,
                use_radii=False,
                edge_threshold=0.0,
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
        enabled: Sequence[str] = ("fcc", "hpc", "bcc"),
        rmsd_threshold: float = 0.1,
        only_selected: bool = False,
        compute: bool = True,
        **kwargs,
    ) -> None:
        """Perform Polyhedral template matching.

        https://dx.doi.org/10.1088/0965-0393/24/5/055007
        https://github.com/pmla/polyhedral-template-matching.

        Args:
            enabled (tuple): List of strings for enabled structure types. Possible values:
                fcc-hcp-bcc-ico-sc-dcub-dhex-graphene
            rmsd_threshold (float): RMSD threshold.
            only_selected (bool): Only selected particles.
            compute (bool): Compute results.
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
            **kwargs: Additional arguments for the modifier.

        Returns:
            None
        """
        if isinstance(enabled, str):
            enabled = [enabled]
        for i in enabled:
            if i not in ["fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene"]:
                msg = f"Enabled structure type {i} unknown"
                raise ValueError(msg)
        if self.backend == "ovito":
            from ovito.modifiers import PolyhedralTemplateMatchingModifier

            _ptm = PolyhedralTemplateMatchingModifier(  # type: ignore[call-arg]
                rmsd_cutoff=rmsd_threshold,
                only_selected=only_selected,
                **kwargs,
            )

            # Enabled by default: FCC, HCP, BCC
            if "fcc" not in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.FCC  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "hcp" not in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.HCP  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "bcc" not in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.BCC  # type: ignore[attr-defined]
                ].enabled = False  # type: ignore[misc]
            if "ico" in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.ICO  # type: ignore[attr-defined]
                ].enabled = True  # type: ignore[misc]
            if "sc" in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.SC  # type: ignore[attr-defined]
                ].enabled = True  # type: ignore[misc]
            if "dcub" in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND  # type: ignore[attr-defined]
                ].enabled = True  # type: ignore[misc]
            if "dhex" in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND  # type: ignore[attr-defined]
                ].enabled = True  # type: ignore[misc]
            if "graphene" in enabled:
                _ptm.structures[
                    PolyhedralTemplateMatchingModifier.Type.GRAPHENE  # type: ignore[attr-defined]
                ].enabled = True  # type: ignore[misc]

            self.pipeline.modifiers.append(_ptm)

        elif self.backend == "lammps":
            # https://docs.lammps.org/compute_ptm_atom.html
            n_compute = len([i["style"] for i in self.pylmp.computes if i["style"] == "ptm/atom"])
            enabled_structures = " ".join(enabled)
            self.pylmp.compute(
                f"ptm_{n_compute} all ptm/atom {enabled_structures} {rmsd_threshold}",
            )
        else:
            raise not_implemented(self.backend)

        if compute:
            self.set_analysis()

    def perform_ajm(self, compute: bool = True) -> None:
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
                [i["style"] for i in self.pylmp.computes if i["style"] == "ackland/atom"],
            )
            self.pylmp.compute(f"ackland_{n_compute} all ackland/atom")

        else:
            pass

        if compute:
            self.set_analysis()

    def perform_csp(self, num_neighbors: int = 12, compute: bool = True) -> None:
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
                [i["style"] for i in self.pylmp.computes if i["style"] == "centro/atom"],
            )
            self.pylmp.compute(f"centro_{n_compute} all centro/atom {num_neighbors}")

        if compute:
            self.set_analysis()

    def get_distinct_grains(
        self,
        *args,
        algorithm: str = "GraphClusteringAuto",
        compute: bool = True,
        **kwargs,
    ) -> None:
        """Get distinct grains from the structure.

        Args:
            *args: Arguments specific to the backend (see below)
            algorithm: Algorithm for grain segmentation.
            compute: Compute results.
            **kwargs: Additional arguments for the algorithm.
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
                msg = "Incorrect Grain Segmentation algorithm specified."
                raise ValueError(msg)

            gsm = GrainSegmentationModifier(*args, algorithm=gsm_mode, **kwargs)
            self.pipeline.modifiers.append(gsm)
            if compute:
                self.set_analysis()
            # TODO: Get misorientation plot

    def set_analysis(self) -> None:
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
            msg = "The pymatgen backend does not require setting the analysis."
            raise NotImplementedError(msg)

    def expand_to_non_selected(
        self,
        nearest_n: int | None = None,
        cutoff: float | None = None,
        expansion_base: list | None = None,
        return_type: str = "Identifier",
        return_random: bool = False,
        invert: bool = False,
    ) -> list[int]:
        """Useful method if only_selected was chosen for structural analysis.

        Args:
            nearest_n: Number of nearest neighbors to consider.
            cutoff: Cutoff distance to consider.
            expansion_base: List of atoms to expand from. Must be Indices.
            return_type: Either Identifier or Indices.
            return_random: If True, return a random selection of the non-selected atoms.
            invert: If True, invert the selection.

        Returns:
        gb_non_selected: list of GB atoms that were not in the previously selected group.
        """
        if nearest_n and cutoff:
            msg = "Only one of nearest_n and cutoff can be specified."
            raise ValueError(msg)
        if self.backend == "ovito":
            if return_type not in ["Identifier", "Indices"]:
                msg = "Only Indices and Identifier are possible as return types."
                raise NameError(msg)

            self._invert_selection()
            self.set_analysis()

            finder = get_finder(self.data, cutoff=cutoff, nearest_n=nearest_n)
            if nearest_n:
                from ovito.data import NearestNeighborFinder

            gb_non_selected = []
            # edge = []
            # Obtain a set of bulk (=crystalline) cations
            bulk_atoms = expansion_base or self.get_crystalline_atoms(return_type="Indices")
            bulk_atoms_set = set(bulk_atoms)
            # These are the atoms that haven't been analysed in the structure analysis, i.e. anions
            non_selected = set(np.where(self.data.particles.selection == 1)[0])

            for index in non_selected:
                neighbors = {neigh.index for neigh in finder.find(index)}
                # The following is the neighbors w/o the atoms excluded from structural analysis
                neighbors_no_selected = neighbors - non_selected
                # If NN, correct for non-selected atoms
                if nearest_n:
                    nearest_n_added = nearest_n
                    while len(neighbors_no_selected) < nearest_n:
                        finder = NearestNeighborFinder(nearest_n_added, self.data)
                        neighbors = {neigh.index for neigh in finder.find(index)}
                        neighbors_no_selected = neighbors - non_selected
                        nearest_n_added += 1
                if len(neighbors_no_selected) == 0:
                    msg = "Cutoff radius too small."
                    raise ValueError(msg)
                if len(neighbors_no_selected) <= 2:
                    warnings.warn(
                        "At least one atoms has only two other atoms to assign. "
                        "Consider increasing the cutoff value.",
                        stacklevel=2,
                    )
                bulk_neighbors = bulk_atoms_set.intersection(neighbors_no_selected)
                bulk_fraction = len(bulk_neighbors) / len(neighbors_no_selected)
                if bulk_fraction < 0.5:
                    gb_non_selected.append(index)
                if return_random and bulk_fraction == 0.5 and random.random() < 0.5:  # noqa: S311
                    gb_non_selected.append(index)

            self._invert_selection()

            if invert:
                gb_non_selected = list(set(non_selected) - set(gb_non_selected))
            if return_type == "Identifier":
                gb_non_selected = [
                    self.data.particles["Particle Identifier"][i] for i in gb_non_selected
                ]
        else:
            raise not_implemented(self.backend)

        return gb_non_selected

    def expand_to_non_selected_groups(
        self,
        groups: list,
        cutoff: float = 4.5,
        return_type: str = "Identifier",
        return_random: bool = False,
    ) -> list:
        """Useful method if only_selected was chosen for structural analysis.

        Args:
            groups: list of lists containing the groups (as indices).
            cutoff (float): Cutoff (in Angstrom) for the neighbour finder.
            return_type (str): return either identifiers or indices.
            return_random (bool): Some particles will have the same (maximum) neighbours in
                multiple groups. If true, returns a random group from that pool.

        Returns:
        groups_non_selected (list): atoms that were not in the previously selected group.
        """
        if self.backend == "ovito":
            if return_type not in ["Identifier", "Indices"]:
                msg = "Only Indices and Identifier are possible as return types."
                raise NameError(msg)

            self._invert_selection()
            self.set_analysis()

            finder = get_finder(self.data, cutoff=cutoff)

            groups_non_selected = [[] for _ in range(len(groups))]  # type: list[list]
            # Obtain sets of bulk (=crystalline) cations
            group_sets = [set(i) for i in groups]
            # These are the atoms that haven't been analysed in the structure analysis, most likely
            # anions
            non_selected = set(np.where(self.data.particles.selection == 1)[0])

            for index in non_selected:
                neighbors = {neigh.index for neigh in finder.find(index)}
                # The following is the neighbors w/o the atoms excluded from structural analysis
                neighbors_no_selected = neighbors - non_selected
                if len(neighbors_no_selected) < 3:
                    warnings.warn(
                        "At least one atoms has only two other atoms to assign.",
                        stacklevel=2,
                    )
                group_neighbors = [len(i.intersection(neighbors_no_selected)) for i in group_sets]
                indices_max = np.where(group_neighbors == np.amax(group_neighbors))[0]
                if len(indices_max) > 1 and return_random:
                    groups_non_selected[random.choice(indices_max)].append(index)  # noqa: S311
                else:
                    group_maximum_interception = np.argmax(group_neighbors)
                    groups_non_selected[group_maximum_interception].append(index)

            self._invert_selection()

            if return_type == "Identifier":
                groups_non_selected = [
                    self.data.particles["Particle Identifier"][i] for i in groups_non_selected
                ]

        else:
            raise not_implemented(self.backend)

        return groups_non_selected

    # TODO: Rename to particles
    def get_non_crystalline_atoms(self, mode: str = "cna", return_type: str = "Identifier") -> list:
        """Get the atoms at the grain boundary.

        For this to work, some sort of structural analysis has to be performed.

        Args:
            mode: Mode for selection of grain boundary atoms.
            return_type (str): Identifier or Indices.

        Returns:
            List of non-crystalline particles.
        """
        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                if return_type == "Identifier":
                    gb_list = [
                        i[0]
                        for i in zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                        if i[1] == 0
                    ]
                elif return_type == "Indices":
                    gb_list = list(np.where(self.data.particles["Structure Type"] == 0)[0])
                else:
                    msg = "Only Indices and Identifier are possible as return types."
                    raise NameError(msg)
            elif "Centrosymmetry" in self.data.particles.keys():
                msg = "Implementation in progress."
                raise NotImplementedError(msg)
                gb_list = []
            else:
                raise not_implemented(self.backend)
        elif self.backend == "lammps":
            # Supported analysis methods: cna, ptm,
            from lammps import LMP_STYLE_ATOM, LMP_TYPE_VECTOR

            # ids = []
            # for i in range(len(self.pylmp.atoms)):
            #     ids.append(self.pylmp.atoms[i].id)
            types = np.concatenate(
                self.pylmp.lmp.numpy.extract_compute("cna_0", LMP_STYLE_ATOM, LMP_TYPE_VECTOR),
            )
            # https://docs.lammps.org/Classes_atom.html#_CPPv4N9LAMMPS_NS4Atom7extractEPKc
            ids = np.concatenate(self.pylmp.lmp.numpy.extract_atom("id"))
            df_temp = pd.DataFrame(
                list(
                    zip(
                        ids,
                        types,
                    ),
                ),
                columns=["Particle Identifier", "Structure Type"],
            )
            # TDOD: This is only cna, what about others?
            if mode == "cna":
                df_gb = df_temp[df_temp["Structure Type"] == 5]
            elif mode in ("ptm", "ackland"):
                df_gb = df_temp[df_temp["Structure Type"] == 0]
            elif mode in ("voronoi", "centro"):
                msg = f"Mode {mode} currently not implemented"
                raise NotImplementedError(msg)
            else:
                msg = f"Incorrect mode {mode} specified"
                raise ValueError(msg)
            gb_list = list(df_gb["Particle Identifier"])
        else:
            raise not_implemented(self.backend)
        return gb_list

    # TODO: Rename to particles
    def get_crystalline_atoms(self, return_type: str = "Identifier") -> list:
        """Get the atoms in the bulk, as determined by structural analysis.

        Returns:
            List of crystalline particles.
        """
        if self.backend == "ovito":
            if "Structure Type" in self.data.particles.keys():
                if return_type == "Identifier":
                    gb_list = [
                        i[0]
                        for i in zip(
                            self.data.particles["Particle Identifier"],
                            self.data.particles["Structure Type"],
                        )
                        if i[1] != 0
                    ]
                elif return_type == "Indices":
                    gb_list = list(np.where(self.data.particles["Structure Type"] != 0)[0])
                else:
                    msg = "Indices and Identifier are possible as return types."
                    raise NotImplementedError(msg)
                # df_temp = pd.DataFrame(
                #     list(
                #         zip(
                #             self.data.particles["Particle Identifier"],
                #             self.data.particles["Structure Type"],
                #         )
                #     ),
                #     columns=["Particle Identifier", "Structure Type"],
                # )
                # df_gb = df_temp[df_temp["Structure Type"] != 0]
                # return list(df_gb["Particle Identifier"])
            else:
                warnings.warn(
                    "No structure type information found. Returning empty list.",
                    stacklevel=2,
                )
                gb_list = []
        elif self.backend == "lammps":
            # TODO: Implement
            raise not_implemented(self.backend)
        else:
            raise not_implemented(self.backend)
        return gb_list

    # TODO: Rename to particles
    def get_grain_edge_ions(
        self,
        nearest_n: int = 12,
        cutoff: float | None = None,
        gb_ions: set | None = None,
        bulk_ions: list | None = None,
        return_type: str = "Identifier",
    ) -> list:
        """Get the atoms at the grain edge, as determined by structural analysis.

        Returns a list of IDs, which were identified as crystalline/bulk atoms, but border at
        least one non-crystalline/grain boundary atom.

        Args:
            nearest_n (int): Number of nearest neighbors to consider. Examples: fcc=12, bcc=8
            cutoff (float): Cutoff distance for the neighbor finder.
            gb_ions (set): Indices of grain boundary ions. Default: non-crystalline ions.
            bulk_ions (list): Indices of bulk ions. Default: crystalline ions.
            return_type (str): Identifier or Indices.

        """
        if self.backend == "ovito":
            # finder: CutoffNeighborFinder | NearestNeighborFinder

            from ovito.data import CutoffNeighborFinder, NearestNeighborFinder

            finder: CutoffNeighborFinder | NearestNeighborFinder
            if cutoff:
                finder = CutoffNeighborFinder(cutoff, self.data)
            else:
                finder = NearestNeighborFinder(nearest_n, self.data)
            # ptypes = self.data.particles.particle_types

            gb_edge_ions = []
            gb_ions_set = gb_ions or self.get_non_crystalline_atoms(return_type="Indices")
            bulk_ions_list = bulk_ions or self.get_crystalline_atoms(return_type="Indices")
            gb_ions_set = set(gb_ions_set)
            for index in bulk_ions_list:
                # print("Nearest neighbors of particle %i:" % index)
                # for neigh in finder.find(index):
                #    print(neigh.index, neigh.distance, neigh.delta)
                #    # The index can be used to access properties of the current neighbor, e.g.
                #    type_of_neighbor = ptypes[neigh.index]
                neighbors = [neigh.index for neigh in finder.find(index)]
                if any(x in gb_ions_set for x in neighbors):
                    gb_edge_ions.append(index)
            if return_type == "Identifier":
                gb_edge_ions = [self.data.particles["Particle Identifier"][i] for i in gb_edge_ions]
        elif self.backend == "lammps":
            # TODO: Implement
            raise not_implemented(self.backend)
        else:
            raise not_implemented(self.backend)
        return gb_edge_ions

    def set_gb_type(self) -> None:
        """Set a property for grain boundary/bulk/grain edge atoms."""

    def get_gb_fraction(self, mode: str = "cna") -> float:
        """Get fraction of grain boundary ions.

        Args:
            mode (str): Mode for selection of grain boundary atoms.

        Returns:
            fraction (float): Fraction of grain boundary ions.
        """
        if self.backend == "ovito":
            fraction = len(self.get_non_crystalline_atoms(mode)) / len(
                self.data.particles["Particle Identifier"],
            )
            warnings.warn(
                "Using all particles with a particle identifier as the base.",
                stacklevel=2,
            )
        elif self.backend == "lammps":
            # TODO: Implement
            raise not_implemented(self.backend)
        else:
            raise not_implemented(self.backend)
        return fraction

    # TODO: Rename to particles
    def get_type(self, atom_type: int, return_type: str = "Identifier") -> list:
        """Get all atoms by type.

        Args:
            atom_type (int): Type of atom.
            return_type (str): Return type ("Identifier" or "Indices")

        Returns:
            List of particles of the specified type.
        """
        if self.backend == "ovito":
            # Currently doesn't work!
            # def assign_particle_types(frame, data):
            #     atom_types = data.particles_.particle_types_
            #
            # self.pipeline.modifiers.append(assign_particle_types)
            # self.set_analysis()
            if return_type == "Identifier":
                atom_list = [
                    i[0]
                    for i in zip(
                        self.data.particles["Particle Identifier"],
                        self.data.particles["Particle Type"],
                    )
                    if i[1] == atom_type
                ]
            elif return_type == "Indices":
                atom_list = list(np.where(self.data.particles["Particle Type"] == atom_type)[0])
            else:
                msg = "Only Indices and Identifier are possible as return types."
                raise NameError(msg)
            # df_temp = pd.DataFrame(
            #     list(
            #         zip(
            #             self.data.particles["Particle Identifier"],
            #             self.data.particles["Particle Type"],
            #         )
            #     ),
            #     columns=["Particle Identifier", "Particle Type"],
            # )
            # df_atom = df_temp[df_temp["Particle Type"].eq(atom_type)]
            # return list(df_atom["Particle Identifier"])

        elif self.backend == "lammps":
            # TODO: Implement
            atom_list = []
        else:
            raise not_implemented(self.backend)
        return atom_list

    # TODO: Verkippungswinkel
    # TODO: Grain Index

    def get_fraction(self, numerator: list, denominator: list) -> float:
        """Get fraction of ions/atoms. Helper function.

        Args:
            numerator: Particle type(s) in the numerator.
            denominator: Particle type(s) in the denominator.

        Returns:
            None
        """
        if self.backend == "ovito":
            num = sum([len(self.get_type(i)) for i in numerator])
            den = sum([len(self.get_type(i)) for i in denominator])
        else:
            raise not_implemented(self.backend)

        return num / den

    def save_image(self, filename: str = "image.png") -> None:
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

    def convert_backend(self, convert_to: available_backends) -> Self:
        """Convert the current backend.

        Args:
            convert_to: Backend to convert to.
        """
        if self.backend == "lammps":
            import datetime

            filename = (
                datetime.datetime.now(tz=datetime.timezone.utc).strftime("%d%m%Y_%H%M%S") + ".lmp"
            )
            self.save_structure("filename", file_type="data")
            if convert_to == "ovito":
                try:
                    return GBStructure(backend=convert_to, filename=filename)  # type: ignore[return-value]
                finally:
                    tempfile = pathlib.Path(filename)
                    tempfile.unlink()
            else:
                raise not_implemented(convert_to)
        else:
            raise not_implemented(self.backend)


class GBStructureTimeseries(GBStructure):
    """This is a class containing multiple snapshots from a time series."""

    # TODO: enable inheritance
    # TODO: get diffusion data
    # TODO: differentiate between along/across GB

    def remove_timesteps(self, timesteps_to_exclude: int) -> None:
        """Remove timesteps from the beginning of a simulation.

        Args:
            timesteps_to_exclude (int): Number of timesteps to exclude

        Returns:
            new_trajectory (:py:class:`polypy.read.Trajectory`):
            Trajectory object.
        """

    # TODO: Add differentiation between diffusion along a grain boundary, transverse to the GB,
    # and between grains


def get_finder(data, cutoff: float | None = None, nearest_n: int | None = None):  # noqa: ANN001,ANN201
    """Get neighbor finder.

    Args:
        data: Data object.
        cutoff: Cutoff distance.
        nearest_n: Number of nearest neighbors.

    Returns:
        finder: Neighbor finder.

    """
    from ovito.data import CutoffNeighborFinder, NearestNeighborFinder

    finder: CutoffNeighborFinder | NearestNeighborFinder

    if cutoff:
        finder = CutoffNeighborFinder(cutoff, data)
    elif nearest_n:
        finder = NearestNeighborFinder(nearest_n, data)
    elif cutoff and nearest_n:
        msg = "Only cutoff or nearest_n can be specified."
        raise NameError(msg)
    else:
        msg = "Either cutoff or nearest_n must be specified."
        raise NameError(msg)
    return finder


def not_implemented(backend: available_backends) -> NotImplementedError:
    """Raise not implemented error.

    Args:
        backend: Backend currently in use.

    Returns:
        NotImplementedError

    """
    return NotImplementedError(f"The backend {backend} doesn't support this function.")

"""Plotting and rendering functions."""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import seaborn as sns
    from matplotlib.figure import Figure
    from ovito.data import DataCollection
    from ovito.pipeline import Pipeline
    from PySide6.QtGui import QImage


def render_ovito(pipeline: Pipeline, res_factor: int = 1) -> QImage:
    """Render an ovito pipeline object.

    Args:
        pipeline: The ovito pipeline to be rendered.
        res_factor: Factor to scale the resolution of the rendering. 2=Full HD, 4=4K.

    Returns:
    -------
        image: Image object. Can be saved via image.save("figure.png")

    """
    from ovito.vis import TachyonRenderer, Viewport  # noqa: PLC0415

    pipeline.add_to_scene()

    viewport = Viewport()
    viewport.type = Viewport.Type.Perspective  # type: ignore[misc]
    viewport.camera_dir = (-1, 2, -1)  # type: ignore[misc]
    viewport.zoom_all(size=(640, 480))

    tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1)  # type: ignore[call-arg]
    return viewport.render_image(
        size=(res_factor * 640, res_factor * 480),
        # filename="figure.png",
        background=(1, 1, 1),
        alpha=True,
        renderer=tachyon,
        crop=True,
    )


def plot_face_order(data: DataCollection, plot_property: str = "Max Face Order") -> sns.FacetGrid:
    """Plot the histogram of max. face order from ovito data.

    Args:
        data (DataCollection): Ovito data collection.
        plot_property (str): Property to be plotted.

    Returns:
        Histogram plot.

    """
    import seaborn as sns  # noqa: PLC0415

    df_temp = pd.DataFrame(
        list(
            zip(
                data.particles["Particle Identifier"],
                data.particles[plot_property],
                strict=True,
            ),
        ),
        columns=["Particle Identifier", plot_property],
    )

    hist_plot = sns.displot(df_temp, x=plot_property, discrete=True)
    return hist_plot.figure


def plot_mdf(
    orientations: np.ndarray,
    bins: int = 30,
    density: bool = True,
) -> Figure:
    """Plot the Misorientation Distribution Function (MDF).

    Computes all unique pairwise misorientation angles from grain quaternion
    orientations and visualises the result as a histogram.

    The misorientation angle between two grains is computed as
    ``2 * arccos(|q1 * q2|)``, where the dot product is taken over all four
    quaternion components.  Because the scalar part of the *relative*
    quaternion ``q1^-1 * q2`` equals ``q1 * q2`` for unit quaternions, the
    formula gives the correct rotation angle regardless of whether the
    quaternions are stored in scalar-first ``(w, x, y, z)`` or scalar-last
    ``(x, y, z, w)`` convention.  Taking the absolute value handles the
    double-cover ambiguity (``q`` and ``-q`` represent the same rotation).

    Args:
        orientations: Grain orientations as unit quaternions, shape ``(N, 4)``.
            Each row represents one grain orientation.  Non-unit quaternions
            are normalised automatically.  At least two orientations are
            required.  When using the ovito backend, orientations are returned
            directly by
            :meth:`~agility.analysis.GBStructure.get_distinct_grains`.
        bins: Number of histogram bins.
        density: If ``True``, normalise the histogram to a probability density.

    Returns:
        matplotlib Figure containing the MDF histogram.

    Raises:
        ValueError: If ``orientations`` does not have shape ``(N, 4)``, if any
            quaternion has zero norm, or if fewer than two orientations are
            provided.

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    q = np.asarray(orientations, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4:
        msg = f"orientations must have shape (N, 4), got {q.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms == 0):
        msg = "orientations contains a zero-norm quaternion"
        raise ValueError(msg)
    q = q / norms

    idx_i, idx_j = np.triu_indices(len(q), k=1)
    if len(idx_i) == 0:
        msg = "at least 2 orientations are required to compute pairwise misorientations"
        raise ValueError(msg)

    # Memory-efficient pairwise dot products: compute only the upper-triangle
    # pairs without materialising the full (N, N) matrix.  Antipodal quaternions
    # (q and -q) are handled by taking the absolute value before arccos.
    dots = np.clip(np.abs(np.einsum("ij,ij->i", q[idx_i], q[idx_j])), 0.0, 1.0)
    angles_deg = np.degrees(2.0 * np.arccos(dots))

    fig, ax = plt.subplots()
    ax.hist(angles_deg, bins=bins, density=density, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Misorientation Angle (°)")
    ax.set_ylabel("Probability Density" if density else "Count")
    ax.set_title("Misorientation Distribution Function")
    return fig


# TODO @ab5424: Implement RDF calculation
# https://github.com/ab5424/agility/issues/185
# https://rdfpy.readthedocs.io/en/latest/introduction_and_examples.html
# #example-rdf-of-a-crystal-structure

"""Plotting and rendering functions."""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import seaborn as sns
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
    from ovito.vis import TachyonRenderer, Viewport

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
    import seaborn as sns

    df_temp = pd.DataFrame(
        list(
            zip(
                data.particles["Particle Identifier"],
                data.particles[plot_property],
            ),
        ),
        columns=["Particle Identifier", plot_property],
    )

    hist_plot = sns.displot(df_temp, x=plot_property, discrete=True)
    return hist_plot.figure


# TODO: Visualize Misorientation distribution function
# https://www.osti.gov/pages/servlets/purl/1657149
# https://mtex-toolbox.github.io/index.html

# TODO: get RDFs https://github.com/by256/rdfpy
# https://rdfpy.readthedocs.io/en/latest/introduction_and_examples.html
# #example-rdf-of-a-crystal-structure

"""
Plotting and rendering funtions.
"""

# Copyright (c) Alexander Bonkowski
# Distributed under the terms of the MIT License
# author: Alexander Bonkowski

import pandas as pd
import seaborn as sns


def render_ovito(pipeline=None, res_factor: int = 1):
    """
    Render an ovito pipeline object.
    Args:
        pipeline: The ovito pipeline to be rendered.
        res_factor: Factor to scale the resolution of the redering. 2=Full HD, 4=4K

    Returns:
        image: Image object. Can be saved via image.save("figure.png")

    """
    from ovito.plugins.PyScript import Viewport
    from ovito.plugins.TachyonPython import TachyonRenderer

    pipeline.add_to_scene()
    viewport = Viewport(type=Viewport.Type.Ortho)
    viewport.type = Viewport.Type.Perspective
    viewport.camera_dir = (-1, 2, -1)
    viewport.zoom_all(size=(640, 480))

    tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1)
    image = viewport.render_image(
        size=(res_factor * 640, res_factor * 480),
        # filename="figure.png",
        background=(1, 1, 1),
        alpha=True,
        renderer=tachyon,
        crop=True,
    )

    return image


def plot_face_order(data=None, plot_property="Max Face Order"):
    """
    Plot the histogram of max. face order from ovito data.
    Args:
        data:

    Returns:

    """
    df_temp = pd.DataFrame(
        list(
            zip(
                data.particles["Particle Identifier"],
                data.particles[plot_property],
            )
        ),
        columns=["Particle Identifier", plot_property],
    )

    hist_plot = sns.displot(df_temp, x=plot_property, discrete=True)
    return hist_plot.fig


# TODO: Visualize Misorientation distribution function
# https://www.osti.gov/pages/servlets/purl/1657149
# https://mtex-toolbox.github.io/index.html

# TODO: get RDFs https://github.com/by256/rdfpy
# https://rdfpy.readthedocs.io/en/latest/introduction_and_examples.html
# #example-rdf-of-a-crystal-structure

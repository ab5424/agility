# Copyright (c) Alexander Bonkowski.
# Distributed under the terms of the MIT License.

"""
Setup.py for agility.
"""

import os
from setuptools import setup

__author__ = "Alexander Bonkowski"
__copyright__ = "Copyright Alexander Bonkowski (2021)"
__version__ = "0.0.1"
__maintainer__ = "Alexander Bonkowski"
__email__ = "alexander.bonkowski@rwth-aachen.de"
__date__ = "29/09/2021"

module_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(module_dir, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()

if __name__ == "__main__":
    setup(
        name="agility",
        version=__version__,
        description="Molecular Dynamics analysis.py",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ab5424/agility",
        author=__author__,
        author_email=__email__,
        license="MIT license",
        packages=["agility"],
        zip_safe=False,
        install_requires=["scipy", "numpy", "pandas"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Development Status :: 1 - Planning",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )

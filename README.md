[![Documentation Status](https://readthedocs.org/projects/agility1/badge/?version=latest)](https://agility1.readthedocs.io/en/latest/?badge=latest)
[![code coverage](https://img.shields.io/codecov/c/gh/ab5424/agility)](https://codecov.io/gh/ab5424/agility)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ab5424/agility/main.svg)](https://results.pre-commit.ci/latest/github/ab5424/agility/main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ab5424/agility/HEAD)
[![DOI](https://zenodo.org/badge/411704224.svg)](https://doi.org/10.5281/zenodo.15662617)

# Agility

**A**tomistic **G**rain Boundary and **I**nterface Uti**lity**. This is a library for pre- and postprocessing polycrystalline and grain-boundary structures to use with atomistic codes, e.g. LAMMPS and VASP. It allows top-level processing of those structures by utilizing established methods to differentiate between bulk and interface regions. This allows to extract static as well as dynamic properties of these structures.

## Implementations

While it is intended that `agility` can be used with different "backends" such as ase, babel, pyiron, and others, the main functionality is (currently) implemented with ovito and LAMMPS.

## Installation

There are different ways to install `agility`. Choose what works best with your workflow.

### From source

To build from source, use

    pip install -r requirements.txt

    python setup.py build

    python setup.py install

### Using `pip`

    pip install agility

### Using `conda`

    conda skeleton pypi agility

    conda build agility

    conda install --use-local agility

## Contributing

Any contributions or even questions about the code are welcome - please use the [Issue Tracker](https://github.com/ab5424/agility/issues) or [Pull Requests](https://github.com/ab5424/agility/pulls).

### Development

The development takes place on the `development` branch. Python 3.9 is the minimum requirement. Some backends (like ovito) currently do not support Python 3.10.

## Documentation

The user documentation will be written in python sphinx. The source files should be
stored in the `doc` directory.

## Run tests

After installation, in the home directory, use

    pytest

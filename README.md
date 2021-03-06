[![Documentation Status](https://readthedocs.org/projects/agility1/badge/?version=latest)](https://agility1.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/ab5424/agility/badge.svg?branch=main)](https://coveralls.io/github/ab5424/agility?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ab5424/agility/HEAD)

# Agility

**A**tomistic **G**rain Boundary and **I**nterface Uti**lity**. This is a library for pre- and postprocessing polycrystalline and grain-boundary structures to use with atomistic codes, e.g. LAMMPS and VASP. It allows top-level processing of those structures by utilizing established methods to differentiate between bulk and interface regions. This allows to extract static as well as dynamic properties of these structures.

## Implementations

While it is intendend that `agility` can be used with different "backends" such as ase, babel, pyiron, and others, the main functionality is (currently) implemented with ovito and LAMMPS.



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

If you use VSCode, you might edit `settings.json` as follows:

  ```json
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100", "--ignore=F841"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.pycodestyleEnabled": false,
  "python.linting.pydocstyleEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
      "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
  }
  ```

## Documentation

The user documentation will be written in python sphinx. The source files should be
stored in the `doc` directory.

## Run tests

After installation, in the home directory, use

```bash
% pytest
```



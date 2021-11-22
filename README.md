# Auto-GaBi

This package is intended for **Auto**mated **G**r**a**in **B**oundary and **I**nterface Analysis.

## Installation

There are different ways to install `autogabi`. Choose what works best with your workflow.

### From source

To build from source, use

    pip install -r requirements.txt

    python setup.py build

    python setup.py install

### Using `pip`

    pip install àutogabi

### Using `conda` 

    conda skeleton pypi autogabi

    conda build autogabi
    
    conda install --use-local autogabi

## Contributing

Any contributions or even questions about the code are welcome - please use the [Issue Tracker](https://github.com/ab5424/Auto-GaBi/issues) or [Pull Requests](https://github.com/ab5424/Auto-GaBi/pulls).

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



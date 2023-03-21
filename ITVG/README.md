# Ion Trap Voltage Generation (ITVG)

## Installation

To install VoltageGeneration (in whatever conda environment
you're in) run the following command
from the folder with setup.py and VoltageGeneration.py in it

```bash
pip install -e .
```

This will install the module from the local directory,
so if you need to edit the source you just edit the local files.

## Development

To develop ITVG, it's most ideal to work in a separate environment.
The following command, run from the top level of this repository,
will create an environment `itvg-dev` with all necessary dependencies:

```bash
conda env create -f environment.yml
```

One should still run `pip install -e .` in this environment.

There is a small test suite, written in Python's unittest framework.
To execute, just run it with Python:

```bash
python VoltageGenerationTest.py
```

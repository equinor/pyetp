![Build Status](https://github.com/equinor/pyetp/actions/workflows/ci.yml/badge.svg?branch=main)
![codecov](https://codecov.io/gh/equinor/pyetp/graph/badge.svg?token=S2XDDKKI8U)
![Python](https://img.shields.io/pypi/pyversions/pyetp)
[![PyPI version](https://badge.fury.io/py/pyetp.svg)](https://badge.fury.io/py/pyetp)
![License](https://img.shields.io/github/license/equinor/pyetp)

# Installing the library
This package is published to PyPI, and can be installed via:
```bash
pip install pyetp
```
The library is tested against Python versions 3.10, 3.11, 3.12 and 3.13.

## Local development
Locally we suggest setting up a virtual environment, and installing the latest
version of pip. Then install the library in editable mode along with the
`dev`-dependency group. That is:
```bash
python -m venv .venv
source .venv/bin/activate
pip install pip --upgrade
pip install -e .
pip install --group dev
```


## Linting and formatting
We use ruff as a linter and formatter. To lint run:
```bash
ruff check
```
To run the formatter do:
```bash
ruff format
```
Or if you just want to check what could have been formatted:
```bash
ruff format --check
```


# RESQML versions
The library is built and tested against RESQML v2.0.1. The spec can be
downloaded
[here](https://publications.opengroup.org/standards/energistics-standards/v231a).

# Generated Python objects from RESQML spec
Under `src/pyetp/resqml_objects` you will find Python objects generated from
the RESQML xml spec.

# Documentation
See `/examples` for 2D grid usage

`tests/test_mesh.py` for Unstructured/structured mesh

# Running the unit tests
We have set up unit tests against a local open-etp-server. To start this server
run:
```bash
docker compose -f tests/compose.yml up [--detach]
```
If you want to re-use the same terminal window you should use the
`--detach`-option, otherwise start a new terminal. We use `pytest` for testing,
which can be run via:
```bash
py.test
```

# This library is under active development and subject to breaking changes

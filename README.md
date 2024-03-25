![Build Status](https://github.com/equinor/pyetp/actions/workflows/ci.yml/badge.svg?branch=main)
![Build Status](https://github.com/equinor/pyetp/actions/workflows/snyk.yml/badge.svg?branch=main)
![codecov](https://codecov.io/gh/equinor/pyetp/graph/badge.svg?token=S2XDDKKI8U)
![Python](https://img.shields.io/pypi/pyversions/pyetp)
[![PyPI version](https://badge.fury.io/py/pyetp.svg)](https://badge.fury.io/py/pyetp)
![License](https://img.shields.io/github/license/equinor/pyetp)
# Install
 `pip install pyetp`

# RESQML versions
The library is build and tested against RESQML v2.2. Spec can be downloaded [here](https://publications.opengroup.org/standards/energistics-standards/v231)

# Generated Python objects from RESQML spec
Under `resqml_objects` you will find Pythons objects generated from RESQML xml spec. It is used to ensure consistence data type is used in RESQML.

# Documentation
See `/examples` for 2D grid usage

`tests/test_mesh.py` for Unstructured/structured mesh

# Tests
### Starting etp-test server
`docker compose -f tests/compose.yml up --detach`
### Running pytest from root folder
`poetry run python -m pytest -rs -v`

# This library is under active development and subject to breaking changes

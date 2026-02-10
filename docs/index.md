# Documentation for pyetp and rddms-io

This library consists of three main components (as three separate namespace packages).


1. An RDDMS-client specifically targetting the OSDU
    [open-etp-server](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server)
    under `rddms-io`.
2. An ETP v1.2 client under `pyetp`.
3. Objects and utilities for working, serializing, and de-serializing RESQML
    v2.0.1 objects under `resqml_objects`.


> The following Energistics (c) products were used in the creation of this work:
> Energistics Transfer Protocol (ETP) v1.2 and RESQML v2.0.1

The documentation for the two Energistics standards used are found here:

- For ETP v1.2 see:
   [https://publications.opengroup.org/standards/energistics-standards/energistics-transfer-protocol/v234](https://publications.opengroup.org/standards/energistics-standards/energistics-transfer-protocol/v234).
- For RESQML v2.0.1 see:
  [https://publications.opengroup.org/v231a](https://publications.opengroup.org/v231a).

## Installation
The library is published to PyPI and can be installed via:
```bash
pip install pyetp
```
This includes all the aforementioned packages.


### Local development
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


### Linting and formatting
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

### Running the unit tests
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


## License
The project is licensed under the terms of the Apache 2.0 license.

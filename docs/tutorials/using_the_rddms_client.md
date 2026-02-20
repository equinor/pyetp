# Using the RDDMS Client

In this tutorial we will upload, inspect, download, and delete the regular
surface model from the tutorial ["Setting up a regular
surface"](set_up_regular_surface.md).

???+ "Nomenclature"

    The _Reservoir Domain Data Management Services_ (RDDMS) is a category under
    the OSDU data platform for working with reservoir related models.
    One of the constituents is the
    [_open-etp-server_](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server)
    which is a server storing and validating RESQML and WITSML models, and
    communicating via the _Energistics Transfer Protocol_ (ETP) v1.2.

    We will use the expressions _ETP server_ and _RDDMS server_ interchangeably
    when refering to the open-etp-server.


## Accessing the ETP server

Access to the
[open-etp-server](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server)
depends on how and where the server is hosted.
This library uses a local ETP server for testing purposes.
In this case the server is open, and no authentication is needed.

??? "Connecting to the local server"

    The local ETP server can be started via the Docker compose file
    [`tests/compose.yml`](https://github.com/equinor/pyetp/blob/main/tests/compose.yml).
    To start it run (assuming that current working directory is the top of the
    `pyetp`-directory):
    ```bash
    docker compose -f tests/compose.yml up
    ```
    The server is then served at `ws://localhost:9100`.
    You can check the server capabilities at
    [`http://localhost:9100/.well-known/etp-server-capabilities?GetVersion=etp12.energistics.org`](http://localhost:9100/.well-known/etp-server-capabilities?GetVersion=etp12.energistics.org).
    There is no need for an access token nor a data partition id for the local
    server.
    See the full compose file below.
    ```yaml
    --8<--
    tests/compose.yml
    --8<--
    ```

When the server is hosted alongside other OSDU Data Platform services as part
of [Microsoft
ADME](https://azure.microsoft.com/products/data-manager-for-energy), access to
the server requires a valid JSON web token (JWT) for authentication and a
`data-partition-id` as the server runs in [`multiple`-partition
mode](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server/-/tree/main?ref_type=heads#partition-modes).
See the documentation for
[`msal`](https://learn.microsoft.com/en-us/entra/msal/python/) on getting an
access token to the ETP server when it is hosted on Azure.
The `data-partition-id` is part of the configuration of the server and must be
communicated alongside ids and potential secrets needed to get a token.


??? "Geting access token from [`msal`](https://learn.microsoft.com/en-us/entra/msal/python/)"

    Below follows an example on how to get an access token using the
    [`msal.PublicClientApplication`](https://learn.microsoft.com/python/api/msal/msal.application.publicclientapplication)-class.
    It is loosely based on the examples from the `msal`-documentation.
    The script assumes the existence of a `.env`-file located alongside it and
    that [`python-dotenv`](https://github.com/theskumar/python-dotenv) is
    installed (`#!bash pip install python-dotenv`).
    ```python
    import msal
    from dotenv import dotenv_values

    env = dotenv_values(".env")

    tenant_id = env["TENANT_ID"]
    client_id = env["CLIENT_ID"]
    scope = env["SCOPE"]

    # RDDMS url and data partition id.
    etp_url = env["RDDMS_URL"]
    data_partition_id = env["DATA_PARTITION_ID"]


    # Get access token using the public client flow.
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
    )

    # Ask the user to log in via the browser.
    result = app.acquire_token_interactive(scopes=[scope])
    if "error" in result:
        raise Exception(f"Unable to get token: {result}")

    # Prepend "Bearer " to the access token.
    access_token = "Bearer " + result["access_token"]
    ```

    The `.env`-file should be on the form:
    ```bash
    TENANT_ID = "..."
    CLIENT_ID = "..."
    SCOPE = "..."

    RDDMS_URL = "wss://..."
    DATA_PARTITION_ID = "..."
    ```

The following example will use the local ETP server, but we will make a note on
where the access token and data partition id should be included.

## Connecting to the ETP server with [`RDDMSClient`][rddms_io.client.RDDMSClient]

To set up a connection we recommend using the
[`rddms_connect`][rddms_io.client.rddms_connect]-class as an _asynchronous
context manager_ (see the documentation for `rddms_connect` on other ways of
setting up a connection).
This ensures that the connection is properly closed after the program leaves
the context manager.


???+ "Asynchronous Python in scripts"

    Both `RDDMSClient` and `ETPClient` uses an asynchronous
    [`websockets`-client](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html#websockets.asyncio.client.ClientConnection)
    and are intended to be used in an asynchronous fashion.
    However, Python does not support using `#!python await`-statements directly
    in a script, unless it is wrapped in an `#!python async def` and called via
    `#!python asyncio.run`.
    This is why the following tutorial is wrapped inside an `#!python async def
     main()`-function that is called by `#!python asyncio.run(main()) at the end.

    In a [Jupyter notebook](https://jupyter.org/), an IPython shell, or
    starting the Python REPL with `#!bash python -m asyncio`, you can avoid the
    wrapper function, and instead use `#!python await` directly.

### Creating a dataspace
We start by importing [`rddms_connect`][rddms_io.client.rddms_connect] and
defining a `main`-function for the asynchronous code.
Next, we set up the `uri` of the RDDMS server.
In this example we use the local ETP server discussed in the previous section.
As such there is no `data_partition_id` and no `access_token` needed.
See the previous section for what these should be set to if the server is set
up in the cloud.

```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py::15
--8<--
```

We use `rddms_connect` as a context manager using `#!python async with` to
connect to the RDDMS server, viz.:
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:12:12
--8<--
    ...
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:17:21
--8<--
```
where `rddms_client` is an instance of
[`RDDMSClient`][rddms_io.client.RDDMSClient].

??? "Use of `...` in examples"

    We use the ellipsis, `...`, to denote omitted code in the examples.

Next, we create a dataspace called `eml:///dataspace('rddms_io/demo')` using
the method
[`RDDMSClient.create_dataspace`][rddms_io.client.RDDMSClient.create_dataspace].

```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:12:12
--8<--
    ...
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:17:17
--8<--
        ...
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:21:29
--8<--
```

We have included dummy values for the [access control lists
(ACL)](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server#osdu-integration),
but they are only needed when the server is integrated with OSDU.
The flag `#!python ignore_if_exists=True` ensures that the program does not
crash if the dataspace already exists.

???+ "ETP v1.2 dataspaces"

    A named dataspace in ETP v1.2 is on the form `eml:///dataspace('{path}')`,
    where `{path}` is a string.
    The open-etp-server adds two [additional
    restrictions](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server#restrictions-added-on-top-of-etp-specifications)
    for the dataspace.

    1. The default dataspace `eml:///` is not supported.
    2. The path is on the form `project/scenario`, and is limited to a single
        separator `/`.

### Running the script
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:-2:
--8<--
```


### Full script
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py
--8<--
```

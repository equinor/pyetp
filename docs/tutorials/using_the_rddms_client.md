# Using the RDDMS Client

In this tutorial we will upload, search, download, and delete the regular
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

### Connecting to the ETP server with [`RDDMSClient`][rddms_io.client.RDDMSClient]

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


## Uploading a regular surface to the ETP server
We start by importing the necesary libraries, and [set up a regular
surface](set_up_regular_surface.md).
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py::39
--8<--
```
Next, we define an `async def main`-function to wrap the asynchronous code.
In this example we use the local ETP server discussed in the previous section
as seen in the `uri` variable.
As such there is no `data_partition_id` and no `access_token` needed.
See the previous section for what these should be set to if the server is set
up in the cloud.
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:42:46
--8<--
```
The variable `dataspace_path = "rddms_io/demo"` corresponds to the full
dataspace uri `eml:///dataspace('rddms_io/demo')`.

???+ "ETP v1.2 dataspaces"

    A named dataspace in ETP v1.2 is on the form `eml:///dataspace('{path}')`,
    where `{path}` is a string.
    The open-etp-server adds two [additional
    restrictions](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server#restrictions-added-on-top-of-etp-specifications)
    for the dataspace.

    1. The default dataspace `eml:///` is not supported.
    2. The path is on the form `project/scenario`, and is limited to a single
        separator `/`.

    All methods in [`RDDMSClient`][rddms_io.client.RDDMSClient] that takes in a
    dataspace uri _will also accept a dataspace path_.


### Connecting to the server and creating a dataspace
We use `rddms_connect` as a context manager using `#!python async with` to
connect to the RDDMS server, viz.:
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:48:52
--8<--
```
where `rddms_client` is an instance of
[`RDDMSClient`][rddms_io.client.RDDMSClient].

Having connected we can create our dataspace using the method
[`RDDMSClient.create_dataspace`][rddms_io.client.RDDMSClient.create_dataspace].

```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:53:60
--8<--
```

We have included dummy values for the [access control lists
(ACL)](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server#osdu-integration),
but they are only needed when the server is integrated with OSDU.
The flag `#!python ignore_if_exists=True` ensures that the program does not
crash if the dataspace already exists.

??? Warning "Altering a dataspace"

    A dataspace, once created, can not be altered directly.
    Instead it must be emptied, deleted, and then re-created if the ACLs have
    been set up incorrectly.


### Uploading the surface
To upload the regular surface we call the method
[`RDDMSClient.upload_model`][rddms_io.client.RDDMSClient.upload_model],
and pass in the `dataspace_path`, a list of the three objects, and a dictionary
(this can be any `#!python dict`-like mapping) where the key is the
`path_in_hdf_file` for the surface array and the array as the value.
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:62:68
--8<--
```

???+ "Multiple writers to the same dataspace"

    The open-etp-server has a limitation where there can only be one writer to
    a dataspace at the same time.
    This is enforced via _transactions_.
    In cases where there is contention of write-access, the optional argument
    `debounce` in
    [`RDDMSClient.upload_model`][rddms_io.client.RDDMSClient.upload_model] can
    be used to have the client wait until the dataspace is free for writing instead
    of crashing.
    Either set `debounce` to a non-zero `#!python float`-value or the boolean
    `#!python True`.
    In the former case this the `#!python float`-value will be the maximum
    number of seconds that the client will wait for a transaction before crashing.
    In the latter the client will not time out, and keep waiting until a
    transaction is available.


### Searching on the ETP server
Searching using ETP can roughly be divided into two kinds, search for
dataspaces and search for data objects.

The search for dataspaces will look for all available dataspaces on the server,
with an optional filter based on time for when the dataspace was last written
to.
The method
[`RDDMSClient.list_dataspaces`][rddms_io.client.RDDMSClient.list_dataspaces]
applies this kind of search.
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:70:70
--8<--
```
An example output using `#!python rich.print` on the results gives:
```
--8<--
examples/tutorials/using_the_rddms_client/dataspaces.txt
--8<--
```
The returned results is a `#!python list` of
[`Dataspace`][energistics.etp.v12.datatypes.object.Dataspace]-objects.
This object is described in section 23.34.10 in the ETP v1.2 standard.

For data objects we can apply more filters and more involved patterns.
To list all data objects under a dataspace use the method
[`RDDMSClient.list_objects_under_dataspace`][rddms_io.client.RDDMSClient.list_objects_under_dataspace].
A useful filter to apply on this method is the `data_object_types`-argument.
This lets you limit the results to only certain kinds of RESQML-objects.
Below is an example where we search for all `obj_Grid2dRepresentation`-objects
under the dataspace.
```python
--8<--
examples/tutorials/using_the_rddms_client/using_the_rddms_client.py:72:75
--8<--
```
Printing the `gri_resources` we get a description of the `gri`-object that we
uploaded.
```
--8<--
examples/tutorials/using_the_rddms_client/gri_resources.txt
--8<--
```
The result is a `#!python list` of
[`Resource`][energistics.etp.v12.datatypes.object.Resource]-objects.





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

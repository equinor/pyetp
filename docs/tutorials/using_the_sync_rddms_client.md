# Using the synchronous RDDMS Client

The ETP protocol is designed to be used for long running connections using
websockets.
However, in many cases it is not practical nor necessary to keep a connection
lying around.
Instead we might do a limited set of tasks, e.g., downloading data, and then we
no longer need the connection.
To ease these shorter workflows, we have set up a synchronous RDDMS client —
called [`RDDMSClientSync`][rddms_io.sync_client.RDDMSClientSync] — that lets
the user call the main functionality from
[`RDDMSClient`][rddms_io.client.RDDMSClient] without having to manually set up
and close the connection.
Furthermore, this also avoids the need for using `async` and `await`.


??? Info

    Even though `RDDMSClientSync` is made synchronous, the methods wrap the
    concurrent `RDDMSClient`.
    As such the "heavy lifting" of the methods are still executed concurrently.

In this tutorial we will repeat what is done in the previous tutorial ["Using
the RDDMS Client"](using_the_rddms_client.md) to make it easier to compare the
two clients.

See ["Accessing the ETP
server"](using_the_rddms_client.md#accessing-the-etp-server) for instructions
on how to use the local open-etp-server or get an access token to a published
instance.
We will in this tutorial also work towards the local server.
Connection towards the server is handled in the background by the
[`RDDMSClientSync`][rddms_io.sync_client.RDDMSClientSync], and there is no need
to use a context manager or similar as for the concurrent
[`RDDMSClient`][rddms_io.client.RDDMSClient].

???+ Warning "Token expiration"

    The [`RDDMSClientSync`][rddms_io.sync_client.RDDMSClientSync] stores the
    access parameters towards the ETP server in the class, and only use them
    when one of its methods is called to establish a connection.
    If the client is used in a long running program, e.g., in a notebook where
    it is set up once and then called much later, the token might expire and
    the client will then be unable to connect to the server.
    In this case you should fetch a new token, and then re-create the client
    using the non-expired token.


## Uploading a regular surface to the ETP server using the synchronous client
This tutorial is more or less a copy of the tutorial ["Uploading a regular
surface to the ETP
server"](using_the_rddms_client.md#uploading-a-regular-surface-to-the-etp-server).
We will therefore make it more brief and avoid any terminal output.

The [regular surface is set up](set_up_regular_surface.md) in exactly the same
way as in the tutorial or the concurrent client:
```python
--8<--
examples/tutorials/using_the_sync_rddms_client/using_the_sync_rddms_client.py::37
--8<--
```
The main difference is that instead of importing
[`rddms_connect`][rddms_io.client.rddms_connect] from `rddms_io` we instead
import [`RDDMSClientSync`][rddms_io.sync_client.RDDMSClientSync] from
`rddms_io` directly.

Notably, we also do not import `#!python asyncio`.
As we do not need use `async` and `await` for the `RDDMSClientSync` we
therefore avoid wrapping the script in an asynchronous function.

### Setting up the client
The connection parameters are given by:
```python
--8<--
examples/tutorials/using_the_sync_rddms_client/using_the_sync_rddms_client.py:39:42
--8<--
```
And we can then create an instance of the client via:
```python
--8<--
examples/tutorials/using_the_sync_rddms_client/using_the_sync_rddms_client.py:44:46
--8<--
```
The parameters are identical to the ones passed to
[`rddms_connect`][rddms_io.client.rddms_connect], but no connection is
established yet.


### Connecting and creating a dataspace
We can now call
[`RDDMSClientSync.create_dataspace`][rddms_io.sync_client.RDDMSClientSync.create_dataspace]
to set up a new dataspace.
The parameters are identical to the ones in the concurrent counterpart,
[`RDDMSClient.create_dataspace`][rddms_io.client.RDDMSClient.create_dataspace].
```python
--8<--
examples/tutorials/using_the_sync_rddms_client/using_the_sync_rddms_client.py:48:55
--8<--
```
??? Info "Inner workings of `RDDMSClientSync`"

    When a method is called on `RDDMSClientSync` will set up a context manager
    via [`rddms_connect`][rddms_io.client.rddms_connect] to establish a
    connection, then call the equally named method on
    [`RDDMSClient`][rddms_io.client.RDDMSClient], and finally close the
    connection when leaving the context manager.
    The pattern goes like this:
    ```python
    class RDDMSClientSync:
        ...
        def <some-method>(self, <parameters>):
            async def <some-method>():
                async with rddms_connect(
                    self.<connection-parameters>
                ) as rddms_client:
                    return await rddms_client.<some-method>(<parameters>)
            return asyncio.run(<some-method>())
    ```


### TODO
- Add listing of the methods in the synchronous client to demonstrate how they
wrap the asynchronous client.
- Add comment on missing parameters when they do not make sense, e.g., the
  `handle_transaction`-flag.

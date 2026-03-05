# RDDMS Client

The class [`RDDMSClient`][rddms_io.client.RDDMSClient] is an asynchronous
client that contains high-level methods for interacting with the
open-etp-server.
To set up a connection use [`rddms_connect`][rddms_io.client.rddms_connect] as
an awaitable, a context manager, or generator returning an instance of
[`RDDMSClient`][rddms_io.client.RDDMSClient].

An alternative is to use
[`RDDMSClientSync`][rddms_io.sync_client.RDDMSClientSync] for a synchronous
alternative to `RDDMSClient`.
This avoids the need for using `await` and `async` at the cost of lower
efficiency, and slighly less flexibility.
However, for many use-cases this will be more than enough.

## Setting up a connection

::: rddms_io.client.rddms_connect

## Client implementation

::: rddms_io.client.RDDMSClient

## Synchronous client implementation
The synchronous client `RDDMSClientSync` is a thin wrapper around
[`RDDMSClient`][rddms_io.client.RDDMSClient]. The main difference is that every
method on the synchronous client will set up a connection towards the server,
call the relevant method, and then tear down the connection. If you find
yourself running multiple methods after one another, it might be beneficial to
use `RDDMSClient` instead.

::: rddms_io.sync_client.RDDMSClientSync

## Data types

Data types returned by [`RDDMSClient`][rddms_io.client.RDDMSClient] in various
methods.

::: rddms_io.data_types

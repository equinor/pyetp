# ETP v1.2 Client

The class [`ETPClient`][pyetp.client.ETPClient] is an asynchronous client
implementation of the Energistics Transfer Protocol (ETP) v1.2 using
[`websockets`](https://github.com/python-websockets/websockets).
To set up a connection use [`etp_connect`][pyetp.client.etp_connect] as an
awaitable, a context manager, or a generator returning instances of the
[`ETPClient`][pyetp.client.ETPClient].

This client is meant as a "raw" ETP v1.2 client.
Use [`RDDMSClient`][rddms_io.client.RDDMSClient] if you want a more high-level
interface to the ETP server.

## Setting up a connection

::: pyetp.client.etp_connect

## Client implementation

::: pyetp.client.ETPClient

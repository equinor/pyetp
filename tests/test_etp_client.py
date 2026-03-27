import asyncio

import pytest
import websockets

from energistics.etp.v12.protocol.dataspace import (
    GetDataspaces,
)
from pyetp import etp_connect
from tests.conftest import (
    etp_server_url,
    skip_decorator,
)


@skip_decorator
@pytest.mark.asyncio
async def test_persistent_connect_ws_closing() -> None:
    counter = 0
    async for etp_client in etp_connect(uri=etp_server_url):
        if counter == 10:
            break

        counter += 1
        await etp_client.ws.close()

    assert counter == 10


@skip_decorator
@pytest.mark.asyncio
async def test_persistent_connect_ws_closing_operations() -> None:
    counter = 0
    async for etp_client in etp_connect(uri=etp_server_url):
        if counter == 10:
            break

        counter += 1

        await etp_client.ws.close(1009)

        with pytest.raises(websockets.ConnectionClosed):
            await etp_client.send_and_recv(GetDataspaces())

    assert counter == 10


@skip_decorator
@pytest.mark.asyncio
async def test_persistent_connect_etp_closing() -> None:
    async for etp_client in etp_connect(uri=etp_server_url):
        break

    assert True


@skip_decorator
@pytest.mark.asyncio
async def test_persistent_connect_broken_receiver_task() -> None:
    counter = 0
    with pytest.raises(asyncio.CancelledError):
        async for etp_client in etp_connect(uri=etp_server_url):
            counter += 1
            etp_client._recvtask.cancel("stop")

            # NOTE: This test can take a variable number of seconds to complete
            # due to the adaptive timeout when waiting for a message.
            await etp_client.send_and_recv(GetDataspaces())

    # Check that the for-loop only iterates once.
    assert counter == 1


@skip_decorator
@pytest.mark.asyncio
async def test_manual_open_close() -> None:
    etp_client = await etp_connect(uri=etp_server_url)
    await etp_client.close()  # close

    with pytest.raises(websockets.ConnectionClosedOK):
        await etp_client.ws.ping()


@skip_decorator
@pytest.mark.asyncio
async def test_disconnect_error() -> None:
    # Websockets closing code 1000 corresponds to a normal closure and
    # websockets closing code 1002 corresponds to an endpoint terminating the
    # connection due to a protocol error (see:
    # https://datatracker.ietf.org/doc/html/rfc6455.html#section-7.4.1).

    with pytest.raises(websockets.exceptions.ConnectionClosedOK):
        async with etp_connect(uri=etp_server_url) as etp_client:
            # Successful closing.
            await etp_client.ws.close(code=1000)
            await etp_client.send_and_recv(GetDataspaces())

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        async with etp_connect(uri=etp_server_url) as etp_client:
            await etp_client.ws.close(code=1002)
            await etp_client.send_and_recv(GetDataspaces())


@skip_decorator
@pytest.mark.asyncio
async def test_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(asyncio.exceptions.TimeoutError):
        # We use a very short timeout to ensure that we don't have to wait a
        # long time for the test to finish.
        async with etp_connect(uri=etp_server_url, etp_timeout=0.1) as etp_client:
            monkeypatch.setattr(asyncio.Event, "set", lambda self: False)
            await etp_client.send_and_recv(GetDataspaces())

import pytest
import pytest_asyncio

from pyetp.client import  connect
from pyetp.client import  ETPClient
from pyetp.config import SETTINGS
from pyetp.uri import  DataspaceURI

SETTINGS.application_name = "geomin_testing"
SETTINGS.etp_url = "ws://localhost:9100"

SETTINGS.dataspace = "testing_space"


async def get_app_token(rc=None):
    return None



@pytest_asyncio.fixture
async def eclient():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ws_open = sock.connect_ex(('127.0.0.1', 9100)) == 0

    if not ws_open:
        pytest.skip(reason="websocket for test server not open", allow_module_level=True)

    async with connect() as client:
        yield client


@pytest_asyncio.fixture
async def default_duri(eclient: ETPClient):
    await eclient.put_dataspaces_no_raise(SETTINGS.duri)
    yield SETTINGS.duri
    await eclient.delete_dataspaces(SETTINGS.duri)



@pytest_asyncio.fixture
async def duri(eclient: ETPClient):
    try:
        uri = DataspaceURI.from_name('test/test')
        resp = await eclient.put_dataspaces_no_raise(uri)
        # assert len(resp) == 1, "created one dataspace"
        yield uri
    finally:
        resp = await eclient.delete_dataspaces(uri)
        assert len(resp) == 1, "should cleanup our test dataspace"

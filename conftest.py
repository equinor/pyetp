
import fakeredis
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from map_api import deps, etp_client
from map_api.config import SETTINGS
from map_api.main import app

ETP_SERVER_URL = "ws://localhost:9002"
SETTINGS.application_name = "geomin_testing"
SETTINGS.etp_url = ETP_SERVER_URL  # type: ignore
SETTINGS.dataspace = "testing_space"


async def get_app_token(rc=None):
    return None


app.dependency_overrides[deps.get_redis] = lambda: fakeredis.aioredis.FakeRedis()
app.dependency_overrides[deps.get_app_token] = get_app_token


@pytest_asyncio.fixture
async def eclient():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ws_open = sock.connect_ex(('127.0.0.1', 9002)) == 0

    if not ws_open:
        pytest.skip(reason="websocket for test server not open", allow_module_level=True)

    async with etp_client.connect(ETP_SERVER_URL) as client:
        yield client


@pytest_asyncio.fixture
async def default_duri(eclient: etp_client.ETPClient):
    await eclient.put_dataspaces_no_raise(SETTINGS.duri)
    yield SETTINGS.duri
    await eclient.delete_dataspaces(SETTINGS.duri)


@pytest.fixture
def client(default_duri):
    return TestClient(app)


@pytest_asyncio.fixture
async def duri(eclient: etp_client.ETPClient):
    try:
        uri = etp_client.DataspaceURI.from_name('test/test')
        resp = await eclient.put_dataspaces_no_raise(uri)
        # assert len(resp) == 1, "created one dataspace"
        yield uri
    finally:
        resp = await eclient.delete_dataspaces(uri)
        assert len(resp) == 1, "should cleanup our test dataspace"

import png
import numpy as np
import pytest
from map_api import tile_service
from fastapi.testclient import TestClient

from map_api.main import TilePostBody, app

client = TestClient(app)


def test_dem():
    grid2d = np.array([[1000.]])
    rgb = tile_service.height2rgb(grid2d)
    assert np.isclose(tile_service.rgb2height(*rgb), 1000.0)


def test_scale():
    assert np.isclose(tile_service.get_scale(0), tile_service.RESOLUTION_LOD0)
    assert np.isclose(tile_service.get_scale(1), tile_service.RESOLUTION_LOD0 / 2)


async def fake_arr_lod(*_):
    return (0, 0), tile_service.empty_tile()


@pytest.mark.parametrize('z', range(3))
@pytest.mark.parametrize('channels', range(1, 5))
def test_api(monkeypatch: pytest.MonkeyPatch, z: int, channels: int):
    monkeypatch.setattr(tile_service, 'get_arr_lod', fake_arr_lod)
    monkeypatch.setattr(tile_service, 'get_tile', lambda *_: tile_service.empty_tile())
    monkeypatch.setattr(tile_service, 'CHANNELS', channels)

    response = client.post(
        f"/tiles/{z}/0/0",
        content=TilePostBody(url='http://localhost', dataspace='testing', rddmsURLs=['http://localhost'] * 3).model_dump_json()
    )

    assert response.status_code == 200, "endpoint should be OK"

    img = png.Reader(bytes=response.content)
    h, w, _, info = img.read()

    assert h == tile_service.TILE_SIZE and w == tile_service.TILE_SIZE, "should return empty tilesize"
    assert info['planes'] == tile_service.CHANNELS, "and return correct number of channels"


@pytest.mark.parametrize('z', range(3))
@pytest.mark.parametrize('channels', range(1, 5))
def test_array_lod(monkeypatch: pytest.MonkeyPatch, z: int, channels: int):
    monkeypatch.setattr(tile_service, 'CHANNELS', channels)

    _, arr = tile_service._get_lod(np.random.random((32, 32)), z, (0, 0), step=(tile_service.RESOLUTION_LOD0, tile_service.RESOLUTION_LOD0))
    assert arr.shape[0] == 32 * 2**z, "should increase with power of 2 per zoom"
    assert arr.shape[2] == channels, "should have correct number of channels"


@pytest.mark.parametrize('z', range(2))
def test_get_tile(monkeypatch: pytest.MonkeyPatch, z: int):
    monkeypatch.setattr(tile_service, 'empty_tile', lambda *_: np.zeros((tile_service.TILE_SIZE, tile_service.TILE_SIZE)).astype(np.uint8))

    # map origo half tile size away project origo ( down right looking on screen )
    arr_ori = -tile_service.TILE_SIZE * np.array([tile_service.RESOLUTION_LOD0] * 2) / 2
    arr = np.ones((tile_service.TILE_SIZE, tile_service.TILE_SIZE)).astype(np.uint8)

    tile = tile_service.get_tile(arr, z, 1, 1, arr_ori)

    if z == 0:
        assert np.isclose(np.sum(tile), np.sum(arr) / 4), "(0, 1, 1) tile should overlap 1/4 of map"
        assert tile[0, 0] == 1 and tile[-1, -1] == 0, "with data located up left corner"
    if z == 1:
        assert np.isclose(np.sum(tile), np.sum(arr)), "(1, 1, 1) tile should overlap perfectly"

    # test invalid location (z, -1, -1)
    with pytest.raises(tile_service.TileError):
        tile_service.get_tile(arr, z, -1, -1, arr_ori)


from contextlib import contextmanager
from typing import Tuple
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
import xtgeo
from conftest import ETP_SERVER_URL
import pyetp.resqml_objects as ro
from pyetp.client import ETPClient, ETPError, connect
from pyetp.types import DataArrayIdentifier
from pyetp.uri import DataspaceURI
from pyetp.utils_xml import (create_epc,
                                          parse_xtgeo_surface_to_resqml_grid)


def create_surface(ncol: int, nrow: int):
    surface = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xori=np.random.rand() * 1000,
        yori=np.random.rand() * 1000,
        xinc=np.random.rand() * 1000,
        yinc=np.random.rand() * 1000,
        values=np.random.random((nrow, ncol)).astype(np.float32),
    )
    return surface


@contextmanager
def temp_maxsize(eclient: ETPClient, maxsize=10000):
    _maxsize_before = eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"]
    try:
        # set maxsize
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = maxsize
        assert eclient.max_size == maxsize
        yield eclient
    finally:
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = _maxsize_before


@pytest_asyncio.fixture
async def uid(eclient: ETPClient, duri: DataspaceURI):
    """empty DataArrayIdentifier"""
    epc_uri, = await eclient.put_resqml_objects(create_epc(), dataspace=duri)
    yield DataArrayIdentifier(uri=str(epc_uri), pathInResource="/data")  # dummy path


@pytest_asyncio.fixture
async def uid_with(eclient: ETPClient, uid: DataArrayIdentifier):
    """DataArrayIdentifier with data already set"""
    data = np.random.rand(100, 50) * 100.
    await eclient.put_array(uid, data.astype(np.float32))
    yield (data, uid)


@pytest.fixture
def uid_not_exists():
    return DataArrayIdentifier(
        uri="eml:///dataspace('never')/eml20.EpcExternalPartReference(5fe90ad4-6d34-4f73-a72d-992b26f8442e)",
        pathInResource='/RESQML/d04327a1-c75c-4961-9b9e-cedfe247511b/zvalues'
    )


#
#
#


@pytest.mark.asyncio
async def test_open_close(monkeypatch: pytest.MonkeyPatch):
    mock_close = AsyncMock()

    async with connect(ETP_SERVER_URL) as client:
        assert client.is_connected, "should be connected"
        await client.close()  # close

        monkeypatch.setattr(client, 'close', mock_close)
    mock_close.assert_called_once()  # ensure close is called on aexit


@pytest.mark.asyncio
async def test_datapaces(eclient: ETPClient, duri: DataspaceURI):
    pass  # basicically just testing if eclient and temp dataspace fixtures


@pytest.mark.asyncio
async def test_arraymeta(eclient: ETPClient, uid_with: Tuple[np.ndarray, DataArrayIdentifier]):
    data, uid = uid_with
    msg = await eclient.get_array_metadata(uid)
    assert len(msg) == 1
    np.testing.assert_allclose(msg[0].dimensions, data.shape)  # type: ignore


@pytest.mark.asyncio
async def test_arraymeta_not_found(eclient: ETPClient, uid_not_exists: DataArrayIdentifier):
    with pytest.raises(ETPError, match="11"):
        await test_arraymeta(eclient, (np.zeros(1, dtype=np.float32), uid_not_exists))


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64, np.bool_])
async def test_get_array(eclient: ETPClient, uid: DataArrayIdentifier, dtype):
    data = np.random.rand(100, 50) * 100.
    data = data.astype(dtype)

    resp = await eclient.put_array(uid, data)
    assert len(resp) == 1

    arr = await eclient.get_array(uid)
    np.testing.assert_allclose(arr, data)

    assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
@pytest.mark.parametrize('shape', [(256, 300), (96, 96, 64)])
async def test_get_array_chuncked(eclient: ETPClient, uid: DataArrayIdentifier, dtype, shape: Tuple[int, ...]):
    data = np.random.rand(*shape) * 100.
    data = data.astype(dtype)  # type: ignore

    resp = await eclient.put_array(uid, data)
    assert len(resp) == 1

    with temp_maxsize(eclient):
        arr = await eclient._get_array_chuncked(uid)
        np.testing.assert_allclose(arr, data)
        assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
async def test_put_array_chuncked(eclient: ETPClient, uid: DataArrayIdentifier, dtype):

    data = np.random.rand(150, 86) * 100.
    data = data.astype(dtype)

    # for some reason writing data takes aloooong time
    eclient.timeout = 60
    with temp_maxsize(eclient):
        await eclient._put_array_chuncked(uid, data)

    arr = await eclient.get_array(uid)
    np.testing.assert_allclose(arr, data)
    assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype', [np.int32, np.float32])  # [np.float32, np.float64, np.int32, np.int64, np.bool_]
@pytest.mark.parametrize('starts', [[0, 0], [20, 20]])  #
async def test_subarrays(eclient: ETPClient, uid: DataArrayIdentifier, dtype, starts):
    data = np.random.rand(100, 50) * 100.
    data = data.astype(dtype)

    resp = await eclient.put_subarray(uid, data, starts=starts, counts=[10, 10], put_uninitialized=True)
    assert len(resp) == 1

    arr = await eclient.get_subarray(uid, starts=starts, counts=[10, 10])
    assert isinstance(arr, np.ndarray)

    assert arr.dtype == dtype

    ends = np.array(starts) + 10
    np.testing.assert_allclose(arr, data[starts[0]:ends[0], starts[0]:ends[1]])


@pytest.mark.asyncio
async def test_resqml_objects(eclient: ETPClient, duri: DataspaceURI):
    surf = create_surface(100, 50)
    epc, crs, gri = parse_xtgeo_surface_to_resqml_grid(surf, 23031)

    uris = await eclient.put_resqml_objects(epc, crs, gri, dataspace=duri)
    epc, crs, gri = await eclient.get_resqml_objects(*uris)

    resp = await eclient.delete_data_objects(*uris)
    assert len(resp) == 3

@pytest.mark.asyncio
@pytest.mark.parametrize('surface', [create_surface(3, 4), create_surface(100, 40)])
async def test_rddms_roundtrip(eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    epc_uri, crs_uri, gri_uri = await eclient.put_xtgeo_surface(surface, epsg_code, dataspace=duri)

    epc, crs, gri = await eclient.get_resqml_objects(epc_uri, crs_uri, gri_uri)
    newsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri)
    array = np.array(newsurf.values.filled(np.nan))

    np.testing.assert_allclose(array, np.array(surface.values.filled(np.nan)))

    print(epc, crs, gri)

    assert isinstance(epc, ro.EpcExternalPartReference)
    assert isinstance(crs, ro.LocalDepth3dCrs)
    assert isinstance(gri, ro.Grid2dRepresentation)

    assert crs.projected_crs.epsg_code == epsg_code
    assert surface.rotation == crs.areal_rotation.value
    assert array.shape[0] == gri.grid2d_patch.slowest_axis_count == surface.values.shape[0] == surface.ncol
    assert array.shape[1] == gri.grid2d_patch.fastest_axis_count == surface.values.shape[1] == surface.nrow

    supporting_geometry = gri.grid2d_patch.geometry.points.supporting_geometry
    assert isinstance(supporting_geometry, ro.Point3dLatticeArray)

    assert surface.xori == supporting_geometry.origin.coordinate1
    assert surface.yori == supporting_geometry.origin.coordinate2
    assert surface.xinc == supporting_geometry.offset[0].spacing.value
    assert surface.yinc == supporting_geometry.offset[1].spacing.value

@pytest.mark.asyncio
async def test_surface(eclient: ETPClient, duri: DataspaceURI):
    surf = create_surface(100, 50)
    epc_uri, _, gri_uri = await eclient.put_xtgeo_surface(surf, dataspace=duri)

    nsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri)
    np.testing.assert_allclose(surf.values, nsurf.values)  # type: ignore

    print(repr(surf))
    print(repr(nsurf))

    # ensure rotation, step, origin etc is equal
    assert surf.generate_hash() == nsurf.generate_hash()

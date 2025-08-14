
import asyncio
import random
import sys
from contextlib import contextmanager
from typing import Tuple
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
import websockets
import xtgeo
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import \
    DataArrayIdentifier
from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_subarrays_type import \
    PutDataSubarraysType
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays import \
    PutDataSubarrays
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays_response import \
    PutDataSubarraysResponse

from pyetp import utils_arrays
import pyetp.resqml_objects as ro
from pyetp.client import ETPClient, ETPError, connect
from pyetp.types import AnyArrayType, DataArrayIdentifier
from pyetp.uri import DataObjectURI, DataspaceURI
from pyetp.utils_arrays import to_data_array, get_transport
from pyetp.utils_xml import (create_epc, instantiate_resqml_grid,
                             parse_xtgeo_surface_to_resqml_grid)


def create_surface(ncol: int, nrow: int, rotation: float):
    surface = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xori=np.random.rand() * 1000,
        yori=np.random.rand() * 1000,
        xinc=np.random.rand() * 1000,
        yinc=np.random.rand() * 1000,
        rotation=rotation,
        values=np.random.random((nrow, ncol)).astype(np.float32),
    )
    return surface


@contextmanager
def temp_maxsize(eclient: ETPClient, maxsize=10000):
    _maxsize_before = eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"]
    try:
        # set maxsize
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = maxsize
        #assert eclient.max_size == maxsize
        yield eclient
    finally:
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = _maxsize_before


@pytest_asyncio.fixture
async def uid(eclient: ETPClient, duri: DataspaceURI):
    """empty DataArrayIdentifier"""
    epc_uri, = await eclient.put_resqml_objects(create_epc(), dataspace_uri=duri)
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

    async with connect() as client:
        assert client.is_connected, "should be connected"
        await client.close()  # close

        monkeypatch.setattr(client, 'close', mock_close)
    mock_close.assert_called_once()  # ensure close is called on aexit


@pytest.mark.asyncio
async def test_manual_open_close():

    client = await connect()
    assert client.is_connected, "should be connected"
    await client.close()  # close

    assert not client.is_connected, "should be disconnected"


@pytest.mark.asyncio
async def test_auth():

    async with connect() as client:
        resp = await client.authorize("test")
        assert resp.success  # test server not protected, so any auth will do


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
async def test_disconnect_error(eclient: ETPClient):

    await eclient.ws.close()

    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await eclient.put_dataspaces_no_raise([""], [""], [""], [""], eclient.dataspace_uri("doesnt matter"))


@pytest.mark.asyncio
async def test_timeout_error(eclient: ETPClient, uid_not_exists: DataArrayIdentifier, monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr(eclient, 'timeout', 0.1)
    monkeypatch.setattr(asyncio.Event, 'set', lambda: None)  # will never signal set

    with pytest.raises(asyncio.exceptions.TimeoutError):
        await eclient.get_array_metadata(uid_not_exists)


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
@pytest.mark.parametrize('shape', [(256, 300), (10, 10, 10)])
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
    await eclient._put_uninitialized_data_array(uid, data.shape, transport_array_type=utils_arrays.get_transport(data.dtype))
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
    transport_array_type = get_transport(data.dtype)
    await eclient._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)
    resp = await eclient.put_subarray(uid, data, starts=starts, counts=[10, 10])
    assert len(resp) == 1

    arr = await eclient.get_subarray(uid, starts=starts, counts=[10, 10])
    assert isinstance(arr, np.ndarray)

    assert arr.dtype == dtype

    ends = np.array(starts) + 10
    np.testing.assert_allclose(arr, data[starts[0]:ends[0], starts[0]:ends[1]])


#@pytest.mark.skip(reason="Regression on test server - enable after bug fix from openetp image")
@pytest.mark.asyncio
async def test_resqml_objects(eclient: ETPClient, duri: DataspaceURI):
    surf = create_surface(100, 50, 0)
    epc, crs, gri = parse_xtgeo_surface_to_resqml_grid(surf, 23031)
    uris = await eclient.put_resqml_objects(epc, crs, gri, dataspace_uri=duri)
    epc, crs, gri = await eclient.get_resqml_objects(*uris)
    resp = await eclient.delete_data_objects(*uris)
    assert len(resp) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', [create_surface(3, 4, 0), create_surface(100, 40, 0)])
async def test_rddms_roundtrip(eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    epc_uri, gri_uri, crs_uri = await eclient.put_xtgeo_surface(surface, epsg_code, duri)
    epc, crs, gri = await eclient.get_resqml_objects(epc_uri, crs_uri, gri_uri)
    newsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
    array = np.array(newsurf.values.filled(np.nan))

    np.testing.assert_allclose(array, np.array(surface.values.filled(np.nan)))

    assert isinstance(epc, ro.EpcExternalPartReference)
    assert isinstance(crs, ro.LocalDepth3dCrs)
    assert isinstance(gri, ro.Grid2dRepresentation)

    assert crs.projected_crs.epsg_code == epsg_code
    assert surface.rotation == crs.areal_rotation.value
    assert array.shape[0] == gri.grid2d_patch.slowest_axis_count == surface.values.shape[0] == surface.ncol
    assert array.shape[1] == gri.grid2d_patch.fastest_axis_count == surface.values.shape[1] == surface.nrow

    supporting_geometry = gri.grid2d_patch.geometry.points.supporting_geometry
    if sys.version_info[1] != 10:
        assert isinstance(supporting_geometry, ro.Point3dLatticeArray)

    assert surface.xori == supporting_geometry.origin.coordinate1
    assert surface.yori == supporting_geometry.origin.coordinate2
    assert surface.xinc == supporting_geometry.offset[0].spacing.value
    assert surface.yinc == supporting_geometry.offset[1].spacing.value


@pytest.mark.asyncio
async def test_surface(eclient: ETPClient, duri: DataspaceURI):
    surf = create_surface(100, 50, 100)
    epc_uri, gri_uri, crs_uri = await eclient.put_xtgeo_surface(surf, 23031, duri)
    nsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
    np.testing.assert_allclose(surf.values, nsurf.values)  # type: ignore
    # ensure rotation, step, origin etc is equal
    compare_surf(surf, nsurf)
    # assert surf.generate_hash() == nsurf.generate_hash()

def compare_surf(surf1: xtgeo.RegularSurface, surf2: xtgeo.RegularSurface):
    m1 = surf1.metadata.get_metadata()
    m2 = surf2.metadata.get_metadata()
    assert m1 == m2


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', [create_surface(100, 40, 0), create_surface(3, 3, 0)])
async def test_get_xy_from_surface(eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    epc_uri, gri_uri, crs_uri = await eclient.put_xtgeo_surface(surface, epsg_code, duri)
    x_ori = surface.xori
    y_ori = surface.yori
    x_max = x_ori + (surface.xinc*surface.ncol)
    y_max = y_ori + (surface.yinc*surface.nrow)
    x = random.uniform(x_ori, x_max)
    y = random.uniform(y_ori, y_max)
    nearest = await eclient.get_surface_value_x_y(epc_uri, gri_uri,crs_uri, x, y, "nearest")
    xtgeo_nearest = surface.get_value_from_xy((x, y), sampling="nearest")
    assert nearest == pytest.approx(xtgeo_nearest)
    linear = await eclient.get_surface_value_x_y(epc_uri, gri_uri,crs_uri, x, y, "bilinear")
    xtgeo_linear = surface.get_value_from_xy((x, y))
    assert linear == pytest.approx(xtgeo_linear)

    # # test x y index fencing
    x_i = x_max - surface.xinc-1
    y_i = y_max - surface.yinc-1

    linear_i = await eclient.get_surface_value_x_y(epc_uri, gri_uri,crs_uri, x_i, y_i, "bilinear")
    xtgeo_linear_i = surface.get_value_from_xy((x_i, y_i))
    assert linear_i == pytest.approx(xtgeo_linear_i, rel=1e-2)

    # test outside map coverage
    x_ii = x_max + 100
    y_ii = y_max + 100
    linear_ii = await eclient.get_surface_value_x_y(epc_uri, gri_uri,crs_uri, x_ii, y_ii, "bilinear")
    assert linear_ii is None


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', [create_surface(100, 40, 0), create_surface(3, 3, 0)])
async def test_sub_array_map(eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI):
    epc, crs, gri = instantiate_resqml_grid("name", 0, surface.xori, surface.yori, surface.xinc, surface.yinc, surface.ncol, surface.nrow, 12345)
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(epc, crs, gri, dataspace_uri=duri)
    transport_array_type = AnyArrayType.ARRAY_OF_DOUBLE
    uid = DataArrayIdentifier(
        uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
        pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file  # type: ignore
    )
    await eclient._put_uninitialized_data_array(uid, (surface.ncol, surface.nrow), transport_array_type=transport_array_type)
    # upload row by row
    v = surface.values.filled(np.nan)
    for i in range(surface.nrow):
        row = v[:, i]

        starts = np.array([0, i], dtype=np.int64)  # len = 2 [x_start_index, y_start_index]
        counts = np.array((surface.ncol, 1), dtype=np.int64)  # len = 2
        values = row.reshape((surface.ncol, 1))
        dataarray = to_data_array(values)
        payload = PutDataSubarraysType(
            uid=uid,
            data=dataarray.data,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )
        response = await eclient.send(
            PutDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutDataSubarraysResponse), "Expected PutDataSubarraysResponse"
        assert len(response.success) == 1, "expected one success"
    # download the surface
    chunked_surface = await eclient.get_xtgeo_surface(epc_uri, gri_uri,crs_uri)
    chunked_surface = np.array(chunked_surface.values.filled(np.nan))

    # upload surface in one go
    epc_uri_control, gri_uri_control, crs_uri_control  = await eclient.put_xtgeo_surface(surface, 23031, duri)
    control_surface = await eclient.get_xtgeo_surface(epc_uri_control, gri_uri_control, crs_uri_control)
    control_surface = np.array(control_surface.values.filled(np.nan))

    assert np.allclose(chunked_surface, control_surface)

import asyncio
import sys
from contextlib import contextmanager
from typing import Tuple
from unittest.mock import AsyncMock

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio
import websockets
import xtgeo
from conftest import construct_2d_resqml_grid_from_array
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_subarrays_type import (
    PutDataSubarraysType,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays import (
    PutDataSubarrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays_response import (
    PutDataSubarraysResponse,
)

import resqml_objects.v201 as ro
from pyetp import utils_arrays
from pyetp.client import ETPClient, ETPError, connect
from pyetp.uri import DataObjectURI, DataspaceURI
from pyetp.utils_xml import (
    instantiate_resqml_grid,
    parse_xtgeo_surface_to_resqml_grid,
)


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
    _maxsize_before = eclient.client_info.endpoint_capabilities[
        "MaxWebSocketMessagePayloadSize"
    ]
    try:
        # set maxsize
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = (
            maxsize
        )
        assert eclient.max_size == maxsize
        yield eclient
    finally:
        eclient.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = (
            _maxsize_before
        )


@pytest_asyncio.fixture
async def uid_with_data(
    eclient: ETPClient,
    duri: DataspaceURI,
    random_2d_resqml_grid: tuple[
        ro.EpcExternalPartReference,
        ro.LocalDepth3dCrs,
        ro.Grid2dRepresentation,
        npt.NDArray[np.float32],
    ],
):
    """DataArrayIdentifier with data already set"""
    epc, crs, gri, data = random_2d_resqml_grid
    assert data.dtype == np.float32
    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )
    await eclient.put_array(uid, data)
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)
    yield (data, uid)


@pytest.fixture
def uid_not_exists():
    return DataArrayIdentifier(
        uri="eml:///dataspace('never')/eml20.EpcExternalPartReference(5fe90ad4-6d34-4f73-a72d-992b26f8442e)",
        pathInResource="/RESQML/d04327a1-c75c-4961-9b9e-cedfe247511b/zvalues",
    )


@pytest.mark.asyncio
async def test_open_close(monkeypatch: pytest.MonkeyPatch):
    mock_close = AsyncMock()

    async with connect() as client:
        assert client.is_connected, "should be connected"
        await client.close()  # close

        monkeypatch.setattr(client, "close", mock_close)
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
async def test_dataspaces(eclient: ETPClient, duri: DataspaceURI):
    response = await eclient.get_dataspaces()
    assert len(response.dataspaces) == 1
    assert str(duri) == response.dataspaces[0].uri


@pytest.mark.asyncio
async def test_arraymeta(
    eclient: ETPClient, uid_with_data: Tuple[np.ndarray, DataArrayIdentifier]
):
    data, uid = uid_with_data
    msg = await eclient.get_array_metadata(uid)
    assert len(msg) == 1
    np.testing.assert_allclose(msg[0].dimensions, data.shape)  # type: ignore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code_and_error",
    [
        (1000, websockets.exceptions.ConnectionClosedOK),
        (1002, websockets.exceptions.ConnectionClosedError),
    ],
)
async def test_disconnect_error(
    eclient: ETPClient,
    code_and_error: tuple[int, websockets.exceptions.ConnectionClosed],
):
    code, error = code_and_error
    # Websockets closing code 1000 corresponds to a normal closure and
    # websockets closing code 1002 corresponds to an endpoint terminating the
    # connection due to a protocol error (see:
    # https://datatracker.ietf.org/doc/html/rfc6455.html#section-7.4.1).
    await eclient.ws.close(code=code)

    with pytest.raises(error):
        await eclient.put_dataspaces_no_raise(
            [""], [""], [""], [""], eclient.dataspace_uri("doesnt matter")
        )


@pytest.mark.asyncio
async def test_timeout_error(
    eclient: ETPClient,
    uid_not_exists: DataArrayIdentifier,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(eclient, "timeout", 0.1)
    # This ensures that the Event flag will never be set to True
    monkeypatch.setattr(asyncio.Event, "set", lambda self: False)

    with pytest.raises(asyncio.exceptions.TimeoutError):
        await eclient.get_array_metadata(uid_not_exists)


@pytest.mark.asyncio
async def test_arraymeta_not_found(
    eclient: ETPClient, uid_not_exists: DataArrayIdentifier
):
    with pytest.raises(ETPError, match="11"):
        await test_arraymeta(eclient, (np.zeros(1, dtype=np.float32), uid_not_exists))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.bool_, np.int8]
)
async def test_get_array(eclient: ETPClient, duri: DataspaceURI, dtype):
    shape = (100, 50)
    scaling = 100.0
    data = (np.random.rand(*shape) * scaling).astype(dtype)
    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )

    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )
    resp = await eclient.put_array(uid, data)
    assert len(resp) == 1
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    arr = await eclient.get_array(uid)

    np.testing.assert_equal(arr, data)
    assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.bool_, np.int8]
)
async def test_download_array(eclient: ETPClient, duri: DataspaceURI, dtype):
    shape = (100, 50)
    scaling = 100.0
    data = (np.random.rand(*shape) * scaling).astype(dtype)
    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    await eclient.upload_array(
        epc_uri=epc_uri,
        path_in_resource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
        data=data,
    )
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    arr = await eclient.download_array(
        epc_uri, gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
    )

    np.testing.assert_equal(arr, data)
    assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize(
    "shape",
    [(256, 302), (150, 55)],
)
async def test_oversized_models(
    eclient: ETPClient,
    duri: DataspaceURI,
    dtype: npt.DTypeLike,
    shape: tuple[int],
) -> None:
    data = (np.random.rand(*shape) * 100.0).astype(dtype)
    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    with temp_maxsize(eclient, maxsize=10_000):
        transaction_uuid = await eclient.start_transaction(
            dataspace_uri=duri, read_only=False
        )
        epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
            epc, crs, gri, dataspace_uri=duri
        )
        await eclient.upload_array(
            epc_uri=epc_uri,
            path_in_resource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
            data=data,
        )
        _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

        ret_epc, ret_crs, ret_gri = await eclient.get_resqml_objects(
            epc_uri,
            crs_uri,
            gri_uri,
        )

        if (sys.version_info.major, sys.version_info.minor) == (3, 10):
            assert crs.vertical_crs.epsg_code == ret_crs.vertical_crs.epsg_code
            assert crs.projected_crs.epsg_code == ret_crs.projected_crs.epsg_code
            assert crs.uuid == ret_crs.uuid
            assert epc.uuid == ret_epc.uuid
            assert gri.uuid == ret_gri.uuid
        else:
            assert ret_epc == epc
            assert ret_crs == crs
            assert ret_gri == gri

        ret_data = await eclient.download_array(
            epc_uri,
            ret_gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
        )

        np.testing.assert_equal(data, ret_data)


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("shape", [(256, 300), (10, 10, 10)])
async def test_get_array_chunked(
    eclient: ETPClient, duri: DataspaceURI, dtype: npt.DTypeLike, shape: Tuple[int, ...]
):
    data = (np.random.rand(*shape) * 100.0).astype(dtype)
    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )
    resp = await eclient.put_array(uid, data)
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)
    assert len(resp) == 1

    with temp_maxsize(eclient):
        arr = await eclient._get_array_chunked(uid)
        np.testing.assert_allclose(arr, data)
        assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
async def test_put_array_chunked(
    eclient: ETPClient, duri: DataspaceURI, dtype: npt.DTypeLike
):
    data = (np.random.rand(150, 86) * 100.0).astype(dtype)
    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )

    logical_array_type, transport_array_type = (
        utils_arrays.get_logical_and_transport_array_types(data.dtype)
    )
    await eclient._put_uninitialized_data_array(
        uid,
        data.shape,
        logical_array_type=logical_array_type,
        transport_array_type=transport_array_type,
    )

    with temp_maxsize(eclient):
        await eclient._put_array_chunked(uid, data)

    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    arr = await eclient.get_array(uid)
    np.testing.assert_allclose(arr, data)
    assert arr.dtype == dtype


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype", [np.int32, np.float32]
)  # [np.float32, np.float64, np.int32, np.int64, np.bool_]
@pytest.mark.parametrize("starts", [[0, 0], [20, 20]])  #
async def test_subarrays(
    eclient: ETPClient, duri: DataspaceURI, dtype: npt.DTypeLike, starts: list[int]
):
    data = (np.random.rand(100, 50) * 100.0).astype(dtype)

    epc, crs, gri, data = construct_2d_resqml_grid_from_array(data)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )
    logical_array_type, transport_array_type = (
        utils_arrays.get_logical_and_transport_array_types(data.dtype)
    )
    await eclient._put_uninitialized_data_array(
        uid,
        data.shape,
        logical_array_type=logical_array_type,
        transport_array_type=transport_array_type,
    )
    resp = await eclient.put_subarray(uid, data, starts=starts, counts=[10, 10])
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)
    assert len(resp) == 1

    arr = await eclient.get_subarray(uid, starts=starts, counts=[10, 10])
    assert isinstance(arr, np.ndarray)

    assert arr.dtype == dtype

    ends = np.array(starts) + 10
    np.testing.assert_allclose(arr, data[starts[0] : ends[0], starts[0] : ends[1]])


@pytest.mark.asyncio
async def test_resqml_objects(eclient: ETPClient, duri: DataspaceURI):
    surf = create_surface(100, 50, 0)
    epc, crs, gri = parse_xtgeo_surface_to_resqml_grid(surf, 23031)
    data = surf.values.filled(np.nan).astype(np.float32)

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        path_in_resource=(
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ),
    )
    resp = await eclient.put_array(uid, data)
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    grr = await eclient.list_objects(duri)
    # Test that both DataspaceURI-objects and strings are supported
    assert grr == await eclient.list_objects(str(duri))
    uris = [r.uri for r in grr.resources]

    assert len(uris) == 5
    assert str(epc_uri) in uris
    assert str(crs_uri) in uris
    assert str(gri_uri) in uris
    act_uri = next(filter(lambda u: "obj_Activity(" in u, uris))
    ate_uri = next(filter(lambda u: "obj_ActivityTemplate" in u, uris))

    epc_r, crs_r, gri_r, act_r, ate_r = await eclient.get_resqml_objects(
        epc_uri, crs_uri, gri_uri, act_uri, ate_uri
    )

    assert epc == epc_r

    import sys

    # The equality check does not work for Python 3.10. We'll leave this check
    # in here before limiting the project to Python 3.11 and up.
    if sys.version_info.minor == 10 and sys.version_info.major == 3:
        assert crs.vertical_crs.epsg_code == crs_r.vertical_crs.epsg_code
        assert crs.projected_crs.epsg_code == crs_r.projected_crs.epsg_code
        assert gri.uuid == gri_r.uuid
    else:
        assert crs == crs_r
        assert gri == gri_r

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )
    # We do not have to delete the ActivityTemplate. This is meant to be reused.
    resp = await eclient.delete_data_objects(epc_uri, crs_uri, gri_uri, act_uri)
    _ = await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    assert len(resp) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "surface", [create_surface(3, 4, 0), create_surface(100, 40, 0)]
)
async def test_rddms_roundtrip(
    eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI
):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array columns,
    # and the second axis for rows.

    epsg_code = 23031
    epc_uri, gri_uri, crs_uri = await eclient.put_xtgeo_surface(
        surface, epsg_code, duri
    )
    epc, crs, gri = await eclient.get_resqml_objects(epc_uri, crs_uri, gri_uri)
    newsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
    array = np.array(newsurf.values.filled(np.nan))

    np.testing.assert_allclose(array, np.array(surface.values.filled(np.nan)))

    assert isinstance(epc, ro.EpcExternalPartReference)
    assert isinstance(crs, ro.LocalDepth3dCrs)
    assert isinstance(gri, ro.Grid2dRepresentation)

    assert crs.projected_crs.epsg_code == epsg_code
    assert surface.rotation == crs.areal_rotation.value
    assert (
        array.shape[0]
        == gri.grid2d_patch.slowest_axis_count
        == surface.values.shape[0]
        == surface.ncol
    )
    assert (
        array.shape[1]
        == gri.grid2d_patch.fastest_axis_count
        == surface.values.shape[1]
        == surface.nrow
    )

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
    assert surf.metadata.get_metadata() == nsurf.metadata.get_metadata()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "surface", [create_surface(100, 40, 0), create_surface(3, 3, 0)]
)
async def test_sub_array_map(
    eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI
):
    epc, crs, gri = instantiate_resqml_grid(
        "name",
        0,
        surface.xori,
        surface.yori,
        surface.xinc,
        surface.yinc,
        surface.ncol,
        surface.nrow,
        12345,
    )

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )

    epc_uri, crs_uri, gri_uri = await eclient.put_resqml_objects(
        epc, crs, gri, dataspace_uri=duri
    )
    uid = DataArrayIdentifier(
        uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
        pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,  # type: ignore
    )
    logical_array_type, transport_array_type = (
        utils_arrays.get_logical_and_transport_array_types(surface.values.dtype)
    )
    await eclient._put_uninitialized_data_array(
        uid,
        (surface.ncol, surface.nrow),
        logical_array_type=logical_array_type,
        transport_array_type=transport_array_type,
    )
    # upload row by row
    v = surface.values.filled(np.nan)
    for i in range(surface.nrow):
        row = v[:, i]

        starts = np.array(
            [0, i], dtype=np.int64
        )  # len = 2 [x_start_index, y_start_index]
        counts = np.array((surface.ncol, 1), dtype=np.int64)  # len = 2
        values = row.reshape((surface.ncol, 1))
        dataarray = utils_arrays.get_etp_data_array_from_numpy(values)
        payload = PutDataSubarraysType(
            uid=uid,
            data=dataarray.data,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )
        response = await eclient.send(
            PutDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutDataSubarraysResponse), (
            "Expected PutDataSubarraysResponse"
        )
        assert len(response.success) == 1, "expected one success"

    await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    # download the surface
    chunked_surface = await eclient.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
    chunked_surface = np.array(chunked_surface.values.filled(np.nan))

    # upload surface in one go
    epc_uri_control, gri_uri_control, crs_uri_control = await eclient.put_xtgeo_surface(
        surface, 23031, duri
    )
    control_surface = await eclient.get_xtgeo_surface(
        epc_uri_control, gri_uri_control, crs_uri_control
    )
    control_surface = np.array(control_surface.values.filled(np.nan))

    assert np.allclose(chunked_surface, control_surface)

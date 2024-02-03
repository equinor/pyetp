import numpy as np
import pytest
import xtgeo

import map_api.etp_client
import map_api.resqml_objects as resqml_objects

from .config import ETP_SERVER_URL, DATASPACE

import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ws_open = sock.connect_ex(('127.0.0.1', 9002)) == 0


def create_surface(ncol: int, nrow: int):
    surface = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xori=np.random.rand() * 1000,
        yori=np.random.rand() * 1000000,
        xinc=23.0,
        yinc=47.3,
        rotation=33.0,
        values=np.random.random((nrow, ncol)).astype(np.float32),
    )
    surface.unrotate()
    return surface


@pytest.mark.asyncio
@pytest.mark.skipif(not ws_open, reason="websocket for test server not open")
@pytest.mark.parametrize('surface', [create_surface(3, 4), create_surface(100, 40)])
async def test_rddms_roundtrip(surface: xtgeo.RegularSurface):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    rddms_uris = await map_api.etp_client.upload_xtgeo_surface_to_rddms(
        surface, "test-surface",   epsg_code,     ETP_SERVER_URL,   DATASPACE,   ""
    )

    assert len(rddms_uris) == 3

    epc, crs, gri, array = await map_api.etp_client.download_resqml_surface(
        rddms_uris, ETP_SERVER_URL, DATASPACE, ""
    )
    np.testing.assert_allclose(array, np.array(surface.values.filled(np.nan)))

    assert isinstance(epc, resqml_objects.EpcExternalPartReference)
    assert isinstance(crs, resqml_objects.LocalDepth3dCrs)
    assert isinstance(gri, resqml_objects.Grid2dRepresentation)

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
    assert (
        surface.xori
        == gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate1
    )
    assert (
        surface.yori
        == gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate2
    )
    assert (
        surface.xinc
        == gri.grid2d_patch.geometry.points.supporting_geometry.offset[
            0
        ].spacing.value
    )
    assert (
        surface.yinc
        == gri.grid2d_patch.geometry.points.supporting_geometry.offset[
            1
        ].spacing.value
    )

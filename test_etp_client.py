import unittest
import unittest.mock
import asyncio

import numpy as np
import xtgeo

import map_api.etp_client
import map_api.resqml_objects as resqml_objects

from .config import ETP_SERVER_URL, DATASPACE


class TestETPClient(unittest.TestCase):
    def test_small_rddms_roundtrip(self):
        # NOTE: xtgeo calls the first axis (axis 0) of the values-array
        # columns, and the second axis by rows.
        ncol, nrow = 3, 4
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

        epsg_code = 23031
        rddms_uris = asyncio.run(
            map_api.etp_client.upload_xtgeo_surface_to_rddms(
                surface=surface,
                title="test-surface",
                projected_epsg=epsg_code,
                etp_server_url=ETP_SERVER_URL,
                dataspace=DATASPACE,
                authorization="",
            )
        )

        assert len(rddms_uris) == 3

        epc, crs, gri, array = asyncio.run(
            map_api.etp_client.download_resqml_surface(
                rddms_uris, ETP_SERVER_URL, DATASPACE, ""
            )
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

    # Mock a smaller MAX_WEBSOCKET_MESSAGE_SIZE to trigger blocking of the arrays
    @unittest.mock.patch(
        "map_api.etp_client.etp_client.MAX_WEBSOCKET_MESSAGE_SIZE", 10000
    )
    def test_subarrays_rddms_roundtrip(self):
        ncol, nrow = 100, 40
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

        epsg_code = 23031
        rddms_uris = asyncio.run(
            map_api.etp_client.upload_xtgeo_surface_to_rddms(
                surface=surface,
                title="test-surface",
                projected_epsg=epsg_code,
                etp_server_url=ETP_SERVER_URL,
                dataspace=DATASPACE,
                authorization="",
            )
        )

        assert len(rddms_uris) == 3

        epc, crs, gri, array = asyncio.run(
            map_api.etp_client.download_resqml_surface(
                rddms_uris, ETP_SERVER_URL, DATASPACE, ""
            )
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

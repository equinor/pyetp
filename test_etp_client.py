import unittest
import asyncio

import numpy as np
import xtgeo

import map_api.etp_client as etp_client

from .config import ETP_SERVER_URL, DATASPACE


class TestETPClient(unittest.TestCase):
    def test_rddms_roundtrip(self):
        ncol, nrow = 3, 4
        surface = xtgeo.RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xori=np.random.rand() * 1000,
            yori=np.random.rand() * 1000000,
            xinc=23.0,
            yinc=47.3,
            rotation=33.0,
            values=np.random.random((nrow, ncol)),
        )
        surface.unrotate()

        rddms_urls = asyncio.run(
            etp_client.upload_xtgeo_surface_to_rddms(
                surface=surface,
                title="test-surface",
                projected_epsg=23031,
                etp_server_url=ETP_SERVER_URL,
                dataspace=DATASPACE,
                authorization="",
            )
        )

        assert len(rddms_urls) == 3

        asyncio.run(
            etp_client.download_xtgeo_surface_from_rddms(
                rddms_urls, ETP_SERVER_URL, DATASPACE, ""
            )
        )

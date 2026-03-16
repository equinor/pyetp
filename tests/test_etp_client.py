import asyncio

import numpy as np
import numpy.typing as npt
import pytest
import websockets

import resqml_objects.v201 as ro
from pyetp import etp_connect, utils_arrays
from pyetp.client import ETPClient, ETPError

from energistics.etp.v12.protocol.dataspace import (
    GetDataspaces,
)

from tests.conftest import (
    skip_decorator,
    etp_server_url,
)


def get_random_surface() -> tuple[
    tuple[
        ro.obj_LocalDepth3dCrs,
        ro.obj_EpcExternalPartReference,
        ro.obj_Grid2dRepresentation,
    ],
    npt.NDArray[np.float64],
]:
    shape = tuple(np.random.randint(10, 123, size=2).tolist())

    x = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[0]
    )
    y = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[1]
    )
    Z = np.exp(
        -(np.linspace(-1, 1, shape[0])[:, None] ** 2)
        - np.linspace(-1, 1, shape[1]) ** 2
    )

    origin = np.array([x[0], y[0]])
    spacing = np.array([x[1] - x[0], y[1] - y[0]])

    grid_angle = 2 * np.pi * (np.random.random() - 0.5)
    grid_unit_vectors = RegularGridParameters.angle_to_unit_vectors(grid_angle)

    crs_angle = 2 * np.pi * (np.random.random() - 0.5)
    crs_offset = 2 * 10 * (np.random.random(2) - 0.5)

    crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(
            title="Random crs",
            originator="rddms-io-tester",
        ),
        vertical_crs=ro.VerticalUnknownCrs(unknown="MSL"),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
        areal_rotation=ro.PlaneAngleMeasure(
            value=crs_angle,
            uom=ro.PlaneAngleUom.RAD,
        ),
        xoffset=float(crs_offset[0]),
        yoffset=float(crs_offset[1]),
    )

    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Random epc", originator="rddms-io-tester"),
    )

    gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=grid_unit_vectors[:, 0],
        unit_vec_2=grid_unit_vectors[:, 1],
    )
    key = gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

    return (epc, crs, gri), {key: Z}


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
        print("HELLO")
        if counter == 10:
            break

        counter += 1

        await etp_client.ws.close(1009)

        with pytest.raises(websockets.ConnectionClosed):
            await etp_client.send_and_recv(GetDataspaces())
        print("FOO?")

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
            etp_client._ETPClient__recvtask.cancel("stop")

            # NOTE: This test can take a variable number of seconds to complete
            # due to the adaptive timeout when waiting for a message.
            await etp_client.send_and_recv(GetDataspaces())

    # Check that the for-loop only iterates once.
    assert counter == 1


@skip_decorator
@pytest.mark.asyncio
async def test_manual_open_close():
    etp_client = await etp_connect(uri=etp_server_url)
    await etp_client.close()  # close

    with pytest.raises(websockets.ConnectionClosedOK):
        await etp_client.ws.ping()

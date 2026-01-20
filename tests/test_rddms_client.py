import numpy as np
import pytest
from conftest import etp_server_url, skip_decorator

import resqml_objects.v201 as ro
from pyetp.client import ETPError
from pyetp.uri import DataspaceURI
from rddms_io.client import rddms_connect
from resqml_objects.surface_helpers import RegularGridParameters


@skip_decorator
@pytest.mark.asyncio
async def test_rddms_connect() -> None:
    async with rddms_connect(uri=etp_server_url) as rddms_client:
        pass

    counter = 0
    async for rddms_client in rddms_connect(uri=etp_server_url):
        if counter == 10:
            break

        counter += 1

    rddms_client = await rddms_connect(uri=etp_server_url)
    await rddms_client.close()


@skip_decorator
@pytest.mark.asyncio
async def test_create_and_delete_dataspaces() -> None:
    async with rddms_connect(uri=etp_server_url) as rddms_client:
        ds_1 = "rddms-io/test-1"
        try:
            await rddms_client.create_dataspace(
                ds_1,
            )
        except ETPError:
            pass

        ds_2 = "rddms-io/test-2"
        try:
            await rddms_client.create_dataspace(
                ds_2,
                legal_tags=["foo"],
                other_relevant_data_countries=["bar"],
                owners=["baz"],
                viewers=["bor"],
            )
        except ETPError:
            pass

        dataspaces = await rddms_client.list_dataspaces()

        assert ds_1 in [d.path for d in dataspaces]
        assert ds_2 in [d.path for d in dataspaces]

        await rddms_client.delete_dataspace(ds_1)
        await rddms_client.delete_dataspace(ds_2)


@skip_decorator
@pytest.mark.asyncio
async def test_upload_and_download_model() -> None:
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

    dataspace_path = "rddms-io/test-upload-and-download-model-2"
    dataspace_uri = str(DataspaceURI.from_any(dataspace_path))

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_path)
        except ETPError:
            pass

        crs_uri, epc_uri, gri_uri = await rddms_client.upload_model(
            dataspace_uri=dataspace_uri,
            ml_objects=[crs, epc, gri],
            data_arrays={
                gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file: Z,
            },
        )

        assert type(crs).__name__ in crs_uri
        assert type(epc).__name__ in epc_uri
        assert type(gri).__name__ in gri_uri

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        (ret_crs, ret_epc, ret_gri), ret_Z = await rddms_client.download_model(
            ml_uris=[crs_uri, epc_uri, gri_uri],
            download_arrays=True,
        )

    assert ret_crs == crs
    assert ret_epc == epc
    assert ret_gri == gri

    np.testing.assert_equal(
        ret_Z[ret_gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file], Z
    )

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        print(await rddms_client.list_objects_under_dataspace(dataspace_uri))
        # await rddms_client.delete_model([crs_uri, epc_uri, gri_uri])
        await rddms_client.delete_dataspace(dataspace_uri)
        print(await rddms_client.list_objects_under_dataspace(dataspace_uri))

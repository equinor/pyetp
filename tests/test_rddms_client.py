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

    dataspace_path = "rddms-io/test-upload-and-download-model"
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

        assert crs_uri == crs.get_etp_data_object_uri(dataspace_uri)
        assert crs_uri == crs.get_etp_data_object_uri(dataspace_path)
        assert epc_uri == epc.get_etp_data_object_uri(dataspace_path)
        assert epc_uri == epc.get_etp_data_object_uri(dataspace_path)
        assert gri_uri == gri.get_etp_data_object_uri(dataspace_path)
        assert gri_uri == gri.get_etp_data_object_uri(dataspace_path)

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
        # List all objects under the dataspace. The server adds two extra
        # objects to the three that we upload, and we need to include these if
        # we wish to delete all the objects under the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        uris = [r.uri for r in resources]

        assert crs_uri in uris
        assert epc_uri in uris
        assert gri_uri in uris

        # Delete all objects under the dataspace.
        await rddms_client.delete_model(uris)

        # Verify that the dataspace is now empty.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        assert len(resources) == 0

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)


@skip_decorator
@pytest.mark.asyncio
async def test_list_linked_objects() -> None:
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

    dataspace_path = "rddms-io/test-list-linked-objects"
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

        gri_lo = await rddms_client.list_linked_objects(
            start_uri=gri_uri,
        )

        assert gri_uri == gri_lo.start_uri

        assert gri_uri in [r.uri for r in gri_lo.source_resources]
        assert crs_uri in [r.uri for r in gri_lo.target_resources]
        assert gri_uri in [e.target_uri for e in gri_lo.source_edges]
        assert gri_uri in [e.source_uri for e in gri_lo.target_edges]
        assert crs_uri in [e.target_uri for e in gri_lo.target_edges]

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # List all objects under the dataspace. The server adds two extra
        # objects to the three that we upload, and we need to include these if
        # we wish to delete all the objects under the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        uris = [r.uri for r in resources]

        assert crs_uri in uris
        assert epc_uri in uris
        assert gri_uri in uris

        # Delete all objects under the dataspace.
        await rddms_client.delete_model(uris)

        # Verify that the dataspace is now empty.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        assert len(resources) == 0

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)


@skip_decorator
@pytest.mark.asyncio
async def test_list_array_metadata() -> None:
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

    gri_1 = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid 1", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=grid_unit_vectors[:, 0],
        unit_vec_2=grid_unit_vectors[:, 1],
    )
    gri_2 = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid 2", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=grid_unit_vectors[:, 0],
        unit_vec_2=grid_unit_vectors[:, 1],
    )

    dataspace_path = "rddms-io/test-list-array-metadata"
    dataspace_uri = str(DataspaceURI.from_any(dataspace_path))

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_path)
        except ETPError:
            pass

        crs_uri, epc_uri, gri_1_uri, gri_2_uri = await rddms_client.upload_model(
            dataspace_uri=dataspace_uri,
            ml_objects=[crs, epc, gri_1, gri_2],
            data_arrays={
                gri_1.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file: Z,
                gri_2.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file: Z,
            },
        )

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        ret_crs, ret_epc, ret_gri_1, ret_gri_2 = await rddms_client.download_model(
            ml_uris=[crs_uri, epc_uri, gri_1_uri, gri_2_uri],
            download_arrays=False,
        )

        assert (
            len(
                await rddms_client.list_array_metadata(
                    dataspace_uri=dataspace_uri,
                    ml_objects=[ret_crs, ret_epc],
                )
            )
            == 0
        )
        metadata = await rddms_client.list_array_metadata(
            dataspace_uri=dataspace_uri,
            ml_objects=[gri_1, gri_2],
        )
        assert gri_1_uri in metadata
        assert gri_2_uri in metadata

        m_1 = metadata[gri_1_uri]
        m_2 = metadata[gri_2_uri]

        pir_1 = gri_1.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        pir_2 = gri_2.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

        assert len(m_1) == len(m_2) == 1
        assert pir_1 in m_1
        assert pir_2 in m_2
        assert tuple(m_1[pir_1].dimensions) == Z.shape
        assert tuple(m_2[pir_2].dimensions) == Z.shape

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # List all objects under the dataspace. The server adds two extra
        # objects to the three that we upload, and we need to include these if
        # we wish to delete all the objects under the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        uris = [r.uri for r in resources]

        assert crs_uri in uris
        assert epc_uri in uris
        assert gri_1_uri in uris
        assert gri_2_uri in uris

        # Delete all objects under the dataspace.
        await rddms_client.delete_model(uris)

        # Verify that the dataspace is now empty.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        assert len(resources) == 0

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)

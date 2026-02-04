import asyncio
import pathlib

import numpy as np
import numpy.typing as npt
import pytest
from conftest import etp_server_url, skip_decorator

import resqml_objects.v201 as ro
from pyetp.client import ETPError
from pyetp.errors import ETPTransactionFailure
from pyetp.uri import DataspaceURI
from pyetp.utils_arrays import get_valid_dtype_cast
from rddms_io.client import rddms_connect
from resqml_objects.epc_readers import (
    get_arrays_and_paths_in_hdf_file,
    get_resqml_v201_objects,
)
from resqml_objects.surface_helpers import RegularGridParameters


def get_random_surface() -> tuple[
    ro.obj_LocalDepth3dCrs,
    ro.obj_EpcExternalPartReference,
    ro.obj_Grid2dRepresentation,
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

    return crs, epc, gri, Z


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
    crs, epc, gri, Z = get_random_surface()

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

    # Test downloading linked objects.
    async with rddms_connect(uri=etp_server_url) as rddms_client:
        ret_objs, ret_Z = await rddms_client.download_model(
            ml_uris=[gri_uri],
            download_arrays=True,
            download_linked_objects=True,
        )

    assert len(ret_objs) == 2
    assert len(ret_Z) == 1

    ret_gri = next(
        filter(lambda o: isinstance(o, ro.obj_Grid2dRepresentation), ret_objs)
    )
    ret_crs = next(filter(lambda o: isinstance(o, ro.obj_LocalDepth3dCrs), ret_objs))

    assert ret_crs == crs
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
    crs, epc, gri, Z = get_random_surface()

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
    crs, epc, gri_1, Z = get_random_surface()

    shape = (
        gri_1.grid2d_patch.slowest_axis_count,
        gri_1.grid2d_patch.fastest_axis_count,
    )

    sg = gri_1.grid2d_patch.geometry.points.supporting_geometry

    origin = np.array([sg.origin.coordinate1, sg.origin.coordinate2])
    spacing = np.array([sg.offset[0].spacing.value, sg.offset[1].spacing.value])
    unit_vec_1 = np.array(
        [sg.offset[0].offset.coordinate1, sg.offset[0].offset.coordinate2]
    )
    unit_vec_2 = np.array(
        [sg.offset[1].offset.coordinate1, sg.offset[1].offset.coordinate2]
    )

    gri_2 = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid 2", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=unit_vec_1,
        unit_vec_2=unit_vec_2,
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
                await rddms_client.list_object_array_metadata(
                    dataspace_uri=dataspace_uri,
                    ml_objects=[ret_crs, ret_epc],
                )
            )
            == 0
        )
        metadata = await rddms_client.list_array_metadata(
            ml_uris=[gri_1_uri, gri_2_uri]
        )
        ometadata = await rddms_client.list_object_array_metadata(
            dataspace_uri=dataspace_uri,
            ml_objects=[gri_1, gri_2],
        )
        assert metadata == ometadata
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


@skip_decorator
@pytest.mark.asyncio
async def test_partial_deletion() -> None:
    crs, epc, gri_1, Z = get_random_surface()

    shape = (
        gri_1.grid2d_patch.slowest_axis_count,
        gri_1.grid2d_patch.fastest_axis_count,
    )

    sg = gri_1.grid2d_patch.geometry.points.supporting_geometry

    origin = np.array([sg.origin.coordinate1, sg.origin.coordinate2])
    spacing = np.array([sg.offset[0].spacing.value, sg.offset[1].spacing.value])
    unit_vec_1 = np.array(
        [sg.offset[0].offset.coordinate1, sg.offset[0].offset.coordinate2]
    )
    unit_vec_2 = np.array(
        [sg.offset[1].offset.coordinate1, sg.offset[1].offset.coordinate2]
    )

    gri_2 = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid 2", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=unit_vec_1,
        unit_vec_2=unit_vec_2,
    )
    gri_1_pir = gri_1.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
    gri_2_pir = gri_2.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

    dataspace_path = "rddms-io/test-partial-deletion"
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
                gri_1_pir: Z,
                gri_2_pir: Z,
            },
        )

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # Confirm that we are unable to delete the local coordinate system (it
        # is referenced by the two grids). Note that we run this test in a
        # separate client as the failed deletion will break the rest of the
        # session.
        with pytest.raises(ETPTransactionFailure):
            await rddms_client.delete_model(
                ml_uris=[crs_uri],
            )

        # NOTE: This test will break if the failed deletion above stops
        # breaking the session. In that case we should move the deletion
        # attempt above to the session below, and remove the
        # `EINVALID_STATE`-check.
        with pytest.raises(ETPError):
            try:
                await rddms_client.list_dataspaces()
            except ETPError as e:
                # ETP error code 8 corresponds to `EINVALID_STATE`
                assert e.code == 8
                raise

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # We start by deleting the `obj_EpcExternalPartReference`-object. This
        # is not referenced by anything as seen by the open-etp-server (it is
        # referenced from the `Hdf5Dataset`-object, but not linked as a target
        # on server).
        await rddms_client.delete_model(
            ml_uris=[epc_uri],
        )

        # Verify that the object has been deleted.
        with pytest.raises(ETPError):
            try:
                await rddms_client.download_model(
                    ml_uris=[epc_uri],
                    download_arrays=False,
                )
            except ETPError as e:
                # An ETP Error code of 11 corresponds to `ENOT_FOUND`.
                assert e.code == 11
                # Re-raise to trigger the test.
                raise

        # Check that we can still download the array (both directly and via the
        # `download_model`-method), even if the `obj_EpcExternalPartReference`
        # is gone.
        gri_1_array = await rddms_client.download_array(
            epc_uri=epc_uri,
            path_in_resource=gri_1_pir,
        )
        np.testing.assert_equal(gri_1_array, Z)
        ret_gri_1, ret_arrays = await rddms_client.download_model(
            ml_uris=[gri_1_uri],
            download_arrays=True,
        )
        np.testing.assert_equal(
            Z,
            ret_arrays[gri_1_pir],
        )

        # Find linked objects to the first grid.
        gri_1_lo = await rddms_client.list_linked_objects(
            start_uri=gri_1_uri,
        )

        # Sort out source uris for the first grid.
        source_uris = [r.uri for r in gri_1_lo.source_resources]
        # Verify that the grid uri is included in the source uris.
        assert gri_1_uri in source_uris
        assert len(source_uris) == 2

        # Delete the first grid and the sources pointing to the grid.
        await rddms_client.delete_model(
            ml_uris=source_uris,
        )
        resources = await rddms_client.list_objects_under_dataspace(dataspace_path)

        resource_uris = [r.uri for r in resources]

        # Verify that all source objects (including the grid) has been deleted.
        assert all([su not in resource_uris for su in source_uris])

        gri_2_lo = await rddms_client.list_linked_objects(
            start_uri=gri_2_uri,
        )

        source_uris = [r.uri for r in gri_2_lo.source_resources]

        # Verify that there are no sources except the second grid itself.
        assert len(source_uris) == 1
        assert gri_2_uri == source_uris[0]

        # Delete the second grid.
        await rddms_client.delete_model(
            ml_uris=source_uris,
        )

        resources = await rddms_client.list_objects_under_dataspace(dataspace_path)

        # Verify that the second grid is no longer stored.
        assert gri_2_uri not in [r.uri for r in resources]

        # Confirm that the array no longer exists on the server.
        with pytest.raises(ETPError):
            try:
                await rddms_client.download_array(
                    epc_uri=epc_uri,
                    path_in_resource=gri_2_pir,
                )
            except ETPError as e:
                # Corresponds to ETP Error code ENOT_FOUND.
                assert e.code == 11
                raise

        crs_lo = await rddms_client.list_linked_objects(
            start_uri=crs_uri,
        )

        source_uris = [r.uri for r in crs_lo.source_resources]

        # Confirm that the crs no longer has any sources attached to it.
        assert len(source_uris) == 1
        assert crs_uri == source_uris[0]

        # Delete the crs.
        await rddms_client.delete_model(
            ml_uris=[crs_uri],
        )

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # Clean-up code. Remove all objects, arrays and the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        await rddms_client.delete_model([r.uri for r in resources])
        # Only the `obj_ActivityTemplate`-object is left.
        assert len(resources) == 1

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)


@skip_decorator
@pytest.mark.asyncio
async def test_debouncing() -> None:
    dataspace_path = "rddms-io/test-debouncing"
    dataspace_uri = str(DataspaceURI.from_any(dataspace_path))

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_path)
        except ETPError:
            pass

    async def task(debounce: bool | float, sleep_time: float) -> None:
        async with rddms_connect(uri=etp_server_url) as rddms_client:
            transaction_uuid = await rddms_client.start_transaction(
                dataspace_uri=dataspace_uri,
                read_only=False,
                debounce=debounce,
            )
            await asyncio.sleep(sleep_time)
            await rddms_client.commit_transaction(transaction_uuid=transaction_uuid)

    # Test a transaction failure when two tasks tries to start a transaction
    # for writing on the same dataspace, and then going to sleep.
    task_1 = asyncio.create_task(task(debounce=False, sleep_time=1))
    task_2 = asyncio.create_task(task(debounce=False, sleep_time=1))

    with pytest.raises(ETPError):
        try:
            # The gather-call will incur an error from one of the tasks, but it
            # will not cancel the remaining task.
            await asyncio.gather(task_1, task_2)
        except ETPError as e:
            # Check that either `task_1` or `task_2` has completed (but not
            # both!).
            assert task_1.done() != task_2.done()

            # Cancel both tasks, one of them should have generated the
            # `ETPError`, and the other is still sleeping and needs to be
            # stopped so that it frees up the dataspace for new transactions.
            task_1.cancel()
            task_2.cancel()
            # Check that we get a `EMAX_TRANSACTIONS_EXCEEDED` error code.
            assert e.code == 15
            raise e

    # Test a transaction failure when the debouncing time is too short.
    task_1 = asyncio.create_task(task(debounce=1.0, sleep_time=10))
    task_2 = asyncio.create_task(task(debounce=1.0, sleep_time=10))

    # This raises an `ETPTransactionFailure` instead of an `ETPError`.
    with pytest.raises(ETPTransactionFailure):
        await asyncio.gather(task_1, task_2)

    # Check that either `task_1` or `task_2` has completed (but not both!).
    assert task_1.done() != task_2.done()

    # Cancel the tasks to free up the dataspace.
    task_1.cancel()
    task_2.cancel()

    # Test working debouncing.
    task_1 = asyncio.create_task(task(debounce=True, sleep_time=1.0))
    task_2 = asyncio.create_task(task(debounce=True, sleep_time=1.0))

    await asyncio.gather(task_1, task_2)

    # Check that both tasks completed successfully.
    assert task_1.done() and task_2.done()

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # Clean-up code. Remove all objects, arrays and the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        await rddms_client.delete_model([r.uri for r in resources])

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)


@skip_decorator
@pytest.mark.asyncio
async def test_debouncing_on_upload() -> None:
    crs_1, epc_1, gri_1, Z_1 = get_random_surface()
    crs_2, epc_2, gri_2, Z_2 = get_random_surface()
    crs_3, epc_3, gri_3, Z_3 = get_random_surface()

    dataspace_path = "rddms-io/test-debouncing-on-upload"
    dataspace_uri = str(DataspaceURI.from_any(dataspace_path))

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_path)
        except ETPError:
            pass

    async def task(
        crs: ro.obj_LocalDepth3dCrs,
        epc: ro.obj_EpcExternalPartReference,
        gri: ro.obj_Grid2dRepresentation,
        Z: npt.NDArray[np.float64],
    ) -> None:
        data_arrays = {
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file: Z,
        }
        async with rddms_connect(uri=etp_server_url) as rddms_client:
            await rddms_client.upload_model(
                dataspace_uri=dataspace_uri,
                ml_objects=[crs, epc, gri],
                data_arrays=data_arrays,
                debounce=True,
            )

    task_1 = asyncio.create_task(task(crs=crs_1, epc=epc_1, gri=gri_1, Z=Z_1))
    task_2 = asyncio.create_task(task(crs=crs_2, epc=epc_2, gri=gri_2, Z=Z_2))
    task_3 = asyncio.create_task(task(crs=crs_3, epc=epc_3, gri=gri_3, Z=Z_3))

    await asyncio.gather(task_1, task_2, task_3)

    assert task_1.done() and task_2.done() and task_3.done()

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # Clean-up code. Remove all objects, arrays and the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        uris = [r.uri for r in resources]

        # Check that all objects were successfully uploaded.
        assert crs_1.get_etp_data_object_uri(dataspace_uri) in uris
        assert crs_2.get_etp_data_object_uri(dataspace_uri) in uris
        assert crs_3.get_etp_data_object_uri(dataspace_uri) in uris

        assert epc_1.get_etp_data_object_uri(dataspace_uri) in uris
        assert epc_2.get_etp_data_object_uri(dataspace_uri) in uris
        assert epc_3.get_etp_data_object_uri(dataspace_uri) in uris

        assert gri_1.get_etp_data_object_uri(dataspace_uri) in uris
        assert gri_2.get_etp_data_object_uri(dataspace_uri) in uris
        assert gri_3.get_etp_data_object_uri(dataspace_uri) in uris

        await rddms_client.delete_model(uris)

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)


@skip_decorator
@pytest.mark.parametrize(
    "input_mesh_file",
    [
        pathlib.Path("data") / "model_hexa_0.epc",
        pathlib.Path("data") / "model_hexa_ts_0_new.epc",
    ],
)
@pytest.mark.asyncio
async def test_epc_file_roundtrip(input_mesh_file: pathlib.Path) -> None:
    ml_objects = get_resqml_v201_objects(input_mesh_file)
    input_hdf_file = input_mesh_file.with_suffix(".h5")
    data_arrays = get_arrays_and_paths_in_hdf_file(input_hdf_file)

    dataspace_path = "rddms-io/test-epc-file-roundtrip"
    dataspace_uri = str(DataspaceURI.from_any(dataspace_path))

    # Cast array data types to valid transport array types. This is needed as
    # the open-etp-server does not currently support the use of the logical
    # array types.
    original_dtypes = {}
    casted_data_arrays = {}
    for k, v in data_arrays.items():
        original_dtypes[k] = v.dtype
        casted_data_arrays[k] = v.astype(get_valid_dtype_cast(v))

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_path)
        except ETPError:
            pass

        ml_uris = await rddms_client.upload_model(
            dataspace_uri=dataspace_uri,
            ml_objects=ml_objects,
            data_arrays=casted_data_arrays,
        )

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        ret_ml_objects, ret_data_arrays = await rddms_client.download_model(
            ml_uris=ml_uris,
            download_arrays=True,
        )

    assert ml_objects == ret_ml_objects
    assert sorted(ret_data_arrays) == sorted(casted_data_arrays)

    casted_ret_data_arrays = {
        k: v.astype(original_dtypes[k]) for k, v in ret_data_arrays.items()
    }

    for k in casted_ret_data_arrays:
        np.testing.assert_equal(casted_ret_data_arrays[k], casted_data_arrays[k])

    async with rddms_connect(uri=etp_server_url) as rddms_client:
        # Clean-up code. Remove all objects, arrays and the dataspace.
        resources = await rddms_client.list_objects_under_dataspace(dataspace_uri)
        uris = [r.uri for r in resources]

        # Delete all data objects.
        await rddms_client.delete_model(uris)

        # Delete the dataspace.
        await rddms_client.delete_dataspace(dataspace_uri)
